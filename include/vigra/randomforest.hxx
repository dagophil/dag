#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <vigra/iteratorfacade.hxx>
#include <queue>
#include <map>
#include <type_traits>

//#include "dagraph.hxx"
#include "jungle.hxx"

namespace vigra
{

namespace detail
{
    template <typename ITER>
    struct IterRange
    {
    public:
        typedef ITER Iter;
        Iter begin;
        Iter end;
    };

    template <typename FEATURETYPE>
    struct Split
    {
    public:
        typedef FEATURETYPE FeatureType;
        size_t feature_index;
        FeatureType thresh;
    };

    /// \brief Draw n samples from [begin, end) with replacement.
    template <typename ITER, typename OUTITER>
    void sample_with_replacement(size_t n, ITER begin, ITER end, OUTITER out)
    {
        size_t num_instances = std::distance(begin, end);
        UniformIntRandomFunctor<MersenneTwister> rand;
        for (size_t i = 0; i < n; ++i)
        {
            size_t k = rand() % num_instances;
            *out = begin[k];
            ++out;
        }
    }

    /// \brief Draw n samples from [begin, end) without replacement.
    template <typename ITER, typename OUTITER>
    void sample_without_replacement(size_t n, ITER begin, ITER end, OUTITER out)
    {
        typedef typename std::iterator_traits<ITER>::value_type value_type;

        size_t num_instances = std::distance(begin, end);
        std::vector<value_type> values(begin, end);
        UniformIntRandomFunctor<MersenneTwister> rand;
        for (size_t i = 0; i < n; ++i)
        {
            size_t k = rand() % (num_instances-i);
            *out = values[k];
            ++out;
            values[k] = values[num_instances-i-1];
        }
    }

    /// \brief Functor for std::accumulate that can be used to add the values of a map.
    template <typename KEY, typename VALUE>
    struct MapValueAdder
    {
    public:
        VALUE operator()(VALUE const & a, std::pair<KEY, VALUE> const & p) const
        {
            return a + p.second;
        }
    };

    /// \brief Compute the gini impurity.
    /// \param labels_left: Label counts of the left child.
    /// \param label_priors: Total label count.
    template <typename COUNTTYPE>
    float gini_impurity(
            std::vector<COUNTTYPE> const & labels_left,
            std::vector<COUNTTYPE> const & label_priors
    ){
        typedef COUNTTYPE CountType;

        CountType const n_total = std::accumulate(
                    label_priors.begin(),
                    label_priors.end(),
                    0
        );
        CountType const n_left = std::accumulate(
                    labels_left.begin(),
                    labels_left.end(),
                    0
        );
        CountType const n_right = n_total - n_left;

        float gini_left = 1;
        float gini_right = 1;
        for (size_t i = 0; i < labels_left.size(); ++i)
        {
            float const p_left = labels_left[i] / static_cast<float>(n_left);
            float const p_right = (label_priors[i] - labels_left[i]) / static_cast<float>(n_right);
            gini_left -= (p_left*p_left);
            gini_right -= (p_right*p_right);
        }
        return n_left*gini_left + n_right*gini_right;
    }

    template <typename ITER>
    class SplitIterator
    {
    public:

        typedef typename std::iterator_traits<ITER>::value_type value_type;

        SplitIterator(ITER const & begin, ITER const & end)
            : current_(begin),
              next_(std::next(begin)),
              end_(end)
        {
            if (current_ == end_)
                next_ = end_;
            while (next_ != end_)
            {
                if (*next_ != *current_)
                    break;
                ++current_;
                ++next_;
            }
        }

        SplitIterator & operator++()
        {
            while (next_ != end_)
            {
                ++current_;
                ++next_;
                if (*next_ != *current_)
                    break;
            }
        }

        value_type operator*() const
        {
            return (*current_ + *next_) / 2;
        }

        bool operator==(SplitIterator const & other)
        {
            return next_ == other.next_;
        }

        bool operator==(ITER const & other)
        {
            return next_ == other;
        }

        bool operator!=(SplitIterator const & other)
        {
            return next_ != other.next_;
        }

        bool operator!=(ITER const & other)
        {
            return next_ != other;
        }

    protected:

        ITER current_;
        ITER next_;
        ITER end_;

    };

//    /// \brief Iterate over in-between values of a container.
//    /// \example Given a vector with the elements {1., 3., 3., 4., 4., 4., 8.}, the iterator would yield {2., 3.5, 6.}.
//    /// \todo Write tests for the iterator.
//    template <typename ITER>
//    class SplitIterator : public ForwardIteratorFacade<SplitIterator<ITER>, typename std::iterator_traits<ITER>::value_type, true>
//    {
//    public:

//        typedef typename std::iterator_traits<ITER>::value_type value_type;

//        SplitIterator(lemon::Invalid const & inv = lemon::INVALID)
//        {}

//        SplitIterator(ITER const & begin, ITER const & end)
//            : current_(begin),
//              next_(std::next(begin)),
//              end_(end)
//        {
//            while (current_ != end_ && next_ != end_)
//            {
//                if (*next_ != *current_)
//                    break;
//                ++current_;
//                ++next_;
//            }
//        }

//        bool operator==(ITER const & other) const
//        {
//            return next_ == other;
//        }

//        bool operator!=(ITER const & other) const
//        {
//            return next_ != other;
//        }

//    private:

//        friend class vigra::IteratorFacadeCoreAccess;

//        bool equal(
//                SplitIterator const & other
//        ) const {
//            return next_ == other.current_;
//        }

//        void increment()
//        {
//            while (next_ != end_)
//            {
//                ++current_;
//                ++next_;
//                if (*next_ != *current_)
//                    break;
//            }
//        }

//        value_type dereference() const
//        {
//            return (*current_ + *next_) / 2;
//        }

//        ITER current_;
//        ITER next_;
//        ITER end_;

//    };

}

/// \brief This class implements operator[] to return the feature vector of the requested instance.
template <typename T>
class FeatureGetter
{
public:
    typedef T value_type;
    typedef value_type & reference;
    typedef value_type const & const_reference;

    FeatureGetter(MultiArrayView<2, T> const & arr, bool presort = false)
        : arr_(arr),
          presort_(presort)
    {
        if (presort_)
        {
            arr_sorted_.reshape(arr_.shape());
            std::vector<size_t> ind(arr_.shape()[0]);
            std::iota(ind.begin(), ind.end(), 0);
            for (size_t j = 0; j < arr_.shape()[1]; ++j)
            {
                std::sort(ind.begin(), ind.end(),
                        [& arr, & j](size_t a, size_t b) {
                            return arr(a, j) < arr(b, j);
                        }
                );
                for (size_t i = 0; i < ind.size(); ++i)
                {
                    arr_sorted_(ind[i], j) = i;
                }
            }
        }
    }

    /// \brief Return the features of instance i.
    MultiArrayView<1, T> instance_features(size_t i)
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the features of instance i (const version).
    MultiArrayView<1, T> const instance_features(size_t i) const
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the i-th feature of all instances.
    MultiArrayView<1, T> get_features(size_t i)
    {
        return arr_.template bind<1>(i);
    }

    /// \brief Return the i-th feature of all instances (const version).
    MultiArrayView<1, T> const get_features(size_t i) const
    {
        return arr_.template bind<1>(i);
    }

    /// \brief Return the number of instances.
    size_t num_instances() const
    {
        return arr_.shape()[0];
    }

    /// \brief Return the number of features.
    size_t num_features() const
    {
        return arr_.shape()[1];
    }

    reference operator()(size_t i, size_t j)
    {
        return arr_(i, j);
    }

    const_reference operator()(size_t i, size_t j) const
    {
        return arr_(i, j);
    }


    struct Counter
    {
        Counter(size_t p_index = 0, size_t p_count = 0)
            : index(p_index),
              count(p_count)
        {}
        size_t index;
        size_t count;
    };


    template <typename ITER>
    void sort(size_t feat, ITER begin, ITER end) const
    {
        if (presort_)
        {
            std::vector<Counter> v(num_instances());
            for (auto it = begin; it != end; ++it)
            {
                auto & item = v[arr_sorted_(*it, feat)];
                item.index = *it;
                ++item.count;
            }
            auto it = begin;
            for (auto const & count : v)
            {
                for (size_t i = 0; i < count.count; ++i)
                {
                    *it = count.index;
                    ++it;
                }
            }
        }
        else
        {
            auto const & arr = arr_;
            std::sort(begin, end,
                    [& arr, & feat](size_t i, size_t j)
                    {
                        return arr(i, feat) < arr(j, feat);
                    }
            );
        }
    }

protected:
    MultiArrayView<2, T> const & arr_;
    bool presort_;
    MultiArray<2, size_t> arr_sorted_;
};

template <typename T>
class LabelGetter
{
public:
    typedef T value_type;
    typedef typename MultiArrayView<1, T>::iterator iterator;
    typedef typename MultiArrayView<1, T>::const_iterator const_iterator;

    LabelGetter(MultiArrayView<1, T> const & arr)
        : arr_(arr)
    {}

    /// \brief Return the label for instance i.
    T & operator[](size_t i)
    {
        return arr_[i];
    }

    /// \brief Return the label for instance i.
    T const & operator[](size_t i) const
    {
        return arr_[i];
    }

    /// \brief Return the number of instances.
    size_t size() const
    {
        return arr_.size();
    }

    /// \brief Return the number of instances.
    size_t num_instances() const
    {
        return arr_.size();
    }

protected:
    MultiArrayView<1, T> const & arr_;
};



class BootstrapSampler
{

public:

    typedef std::vector<size_t>::iterator iterator;
    typedef std::vector<size_t>::const_iterator const_iterator;
    typedef UniformIntRandomFunctor<MersenneTwister> Random;

    /// \brief Create the bootstrap samples.
    explicit BootstrapSampler()
        : rand_()
    {}

    /// \brief Create a bootstrap sample.
    std::vector<size_t> bootstrap_sample(size_t num_instances) const
    {
        std::vector<size_t> v(num_instances);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = rand_() % num_instances;
        return v;
    }

    /// \brief Return all of the given instances (hence do nothing).
    template <typename ITER>
    void split_sample(ITER & begin, ITER & end) const
    {
    }

protected:

    Random rand_;

    std::vector<size_t> instances_;

};



class PurityTermination
{
public:
    template <typename ITER, typename LABELS>
    bool stop(ITER begin, ITER end, LABELS const & labels, typename LABELS::value_type & first_label) const
    {
        if (std::distance(begin, end) < 2)
            return true;

        first_label = labels[*begin];
        ++begin;
        while (begin != end)
        {
            if (labels[*begin] != first_label)
                return false;
            ++begin;
        }
        return true;
    }
};



/// \brief Simple decision tree class.
template <typename FEATURETYPE, typename LABELTYPE>
class DecisionTree0
{
public:

    typedef BinaryTree Tree;
    typedef typename Tree::Node Node;
    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef detail::Split<FeatureType> Split;

    template <typename T>
    using NodeMap = typename Tree::template NodeMap<T>;

    /// \brief Initialize the tree with the given instance indices.
    DecisionTree0() = default;
    DecisionTree0(DecisionTree0 const &) = default;
    DecisionTree0(DecisionTree0 &&) = default;
    ~DecisionTree0() = default;
    DecisionTree0 & operator=(DecisionTree0 const &) = default;
    DecisionTree0 & operator=(DecisionTree0 &&) = default;

    /// \brief Train the decision tree.
    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION>
    void train(
            FEATURES const & data_x,
            LABELS const & data_y
    );

    /// \brief Predict new data using the forest.
    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & test_x,
            LABELS & pred_y
    ) const;

protected:

    /// \brief Split the given node and return the children. If no split was performed, n0 and n1 are set to lemon::INVALID.
    /// \param node: The node that will be split.
    /// \param data_x: The features.
    /// \param data_y: The labels.
    /// \param[out] n0: The first child node.
    /// \param[out] n1: The second child node.
    /// \param label_buffer0: Buffer storage for labels.
    /// \param label_buffer1: Buffer storage for labels.
    template <typename FEATURES, typename LABELS>
    void split(
            Node const & node,
            FEATURES const & data_x,
            LABELS const & data_y,
            Node & n0,
            Node & n1,
            std::vector<size_t> & label_buffer0,
            std::vector<size_t> & label_buffer1
    );

    /// \brief The graph structure.
    Tree tree_;

    /// \brief The node labels that were found in training.
    NodeMap<LabelType> node_labels_;

    /// \brief The split of each node.
    NodeMap<Split> node_splits_;

    /// \brief Whether the node is a left or a right child.
    NodeMap<bool> is_left_node;

private:

    typedef std::vector<size_t>::iterator InstanceIterator;
    typedef detail::IterRange<InstanceIterator> Range;

//    /// \brief Vector with the instance indices.
//    std::vector<size_t> instance_indices_;

    /// \brief The instances of each node (begin and end iterator in the vector instance_indices_).
    NodeMap<Range> instance_ranges_;

};

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION>
void DecisionTree0<FEATURETYPE, LABELTYPE>::train(
        FEATURES const & features,
        LABELS const & labels
){
    // Create the bootstrap indices.
    SAMPLER sampler;
    std::vector<size_t> instance_indices = sampler.bootstrap_sample(labels.size());

    // Create the queue with the nodes to be split and place the root node with all instances inside.
    std::queue<Node> node_queue;
    auto const rootnode = tree_.addNode();
    instance_ranges_[rootnode] = {instance_indices.begin(), instance_indices.end()};
    node_queue.push(rootnode);

    // The buffer vectors are used to count the labels and label priors in each split.
    // They are created here so the vectors dont need to be allocated in each split.
    std::vector<size_t> label_buffer0;
    std::vector<size_t> label_buffer1;

    // Split the nodes.
    while (!node_queue.empty())
    {
        auto const node = node_queue.front();
        node_queue.pop();

        // Draw a random sample of the instances.
        auto instances = instance_ranges_[node];
        sampler.split_sample(instances.begin, instances.end);

        // Check the termination criterion.
        TERMINATION termination_crit;
        LabelType first_label;
        bool do_split = !termination_crit.stop(instances.begin, instances.end, labels, first_label);
        if (do_split)
        {
            // Split the node.
            Node n0, n1;
            split(node, features, labels, n0, n1, label_buffer0, label_buffer1);
            vigra_assert(n0 != lemon::INVALID && n1 != lemon::INVALID, "Error");
            node_queue.push(n0);
            node_queue.push(n1);
        }
        else
        {
            // Make the node terminal.
            node_labels_[node] = first_label;
        }
    }
}

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS>
void DecisionTree0<FEATURETYPE, LABELTYPE>::predict(
        FEATURES const & test_x,
        LABELS & pred_y
) const {

    Node rootnode = tree_.getRoot();
    vigra_assert(tree_.valid(rootnode), "DecisionTree0::predict(): The graph has no root node.");

    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        auto const feats = test_x.instance_features(i);
        Node node = rootnode;

        while (tree_.outDegree(node) > 0)
        {
             auto const & s = node_splits_.at(node);
             if (feats[s.feature_index] < s.thresh)
             {
                 node = tree_.getChild(node, 0);
             }
             else
             {
                 node = tree_.getChild(node, 1);
             }
        }
        pred_y[i] = node_labels_.at(node);
    }
}

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS>
void DecisionTree0<FEATURETYPE, LABELTYPE>::split(
        Node const & node,
        FEATURES const & data_x,
        LABELS const & data_y,
        Node & n0,
        Node & n1,
        std::vector<size_t> & label_buffer0,
        std::vector<size_t> & label_buffer1
){
    auto const inst_begin = instance_ranges_[node].begin;
    auto const inst_end = instance_ranges_[node].end;
    auto const num_instances = std::distance(inst_begin, inst_end);

    // Get a random subset of the features.
    size_t const num_feats = std::ceil(std::sqrt(data_x.num_features()));
    std::vector<size_t> all_feat_indices;
    all_feat_indices.reserve(data_x.num_features());
    for (size_t i = 0; i < data_x.num_features(); ++i)
        all_feat_indices.push_back(i);
    std::vector<size_t> feat_indices;
    feat_indices.reserve(num_feats);
    detail::sample_without_replacement(num_feats, all_feat_indices.begin(), all_feat_indices.end(), std::back_inserter(feat_indices));

    // Compute the prior label count.
    std::fill(label_buffer0.begin(), label_buffer0.end(), 0);
    for (InstanceIterator it(inst_begin); it != inst_end; ++it)
    {
        size_t const l = static_cast<size_t>(data_y[*it]);
        if (l >= label_buffer0.size())
            label_buffer0.resize(l+1);
        ++label_buffer0[l];
    }

    // Find the best split.
    size_t best_feat = 0;
    FeatureType best_split = 0;
    float best_gini = std::numeric_limits<float>::max();
    for (auto const & feat : feat_indices)
    {
        // Sort the instances according to the current feature.
        data_x.sort(feat, inst_begin, inst_end);

        // Clear the label counter.
        std::fill(label_buffer1.begin(), label_buffer1.end(), 0);

        // Compute the gini impurity of each split.
        auto const features = data_x.get_features(feat);
        size_t first_right_index = 0; // index of the first instance that is assigned to the right child
        for (size_t i = 0; i+1 < num_instances; ++i)
        {
            // Compute the split.
            auto const left = features[*(inst_begin+i)];
            auto const right = features[*(inst_begin+i+1)];
            if (left == right)
                continue;
            auto const s = (left+right)/2;

            // Add the new labels to the left child.
            do
            {
                size_t const new_label = static_cast<size_t>(data_y[*(inst_begin+first_right_index)]);
                if (new_label >= label_buffer1.size())
                    label_buffer1.resize(new_label+1);
                ++label_buffer1[new_label];
                ++first_right_index;
            }
            while (features[*(inst_begin+first_right_index)] < s);

            // Compute the gini.
            float const gini = detail::gini_impurity(label_buffer1, label_buffer0);
            if (gini < best_gini)
            {
                best_gini = gini;
                best_split = s;
                best_feat = feat;
            }
        }
    }

    // Separate the data according to the best split.
    auto const best_features = data_x.get_features(best_feat);
    auto const split_iter = std::partition(inst_begin, inst_end,
            [&](size_t instance_index){
                return best_features[instance_index] < best_split;
            }
    );
    n0 = tree_.addNode();
    n1 = tree_.addNode();
    tree_.addArc(node, n0);
    tree_.addArc(node, n1);
    instance_ranges_[n0] = {inst_begin, split_iter};
    instance_ranges_[n1] = {split_iter, inst_end};
    is_left_node[n0] = true;
    is_left_node[n1] = false;
    node_splits_[node] = {best_feat, best_split};
}



/// \brief Random forest class.
template <typename FEATURETYPE, typename LABELTYPE>
class RandomForest0
{
public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef DecisionTree0<FeatureType, LabelType> Tree;

    RandomForest0() = default;
    RandomForest0(RandomForest0 const &) = default;
    RandomForest0(RandomForest0 &&) = default;
    ~RandomForest0() = default;
    RandomForest0 & operator=(RandomForest0 const &) = default;
    RandomForest0 & operator=(RandomForest0 &&) = default;

    /// \brief Train the random forest.
    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION>
    void train(
            FEATURES const & train_x,
            LABELS const & train_y,
            size_t num_trees
    );

    /// \brief Predict new data using the forest.
    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & test_x,
            LABELS & pred_y
    ) const;

protected:

    /// \brief The trees of the forest.
    std::vector<Tree> dtrees_;

};

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION>
void RandomForest0<FEATURETYPE, LABELTYPE>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::train(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "RandomForest0::train(): Wrong label type.");

    // TODO: Use resize and do the sampling inside the tree.
    dtrees_.resize(num_trees);

    for (size_t i = 0; i < num_trees; ++i)
    {
        // TODO: Remove output.
        std::cout << "training tree " << i << std::endl;
        dtrees_[i].train<FEATURES, LABELS, SAMPLER, TERMINATION>(data_x, data_y);
    }
}

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS>
void RandomForest0<FEATURETYPE, LABELTYPE>::predict(
        FEATURES const & test_x,
        LABELS & pred_y
) const {
    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::predict(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "RandomForest0::predict(): Wrong label type.");

    // Let each tree predict all instances.
    MultiArray<2, LabelType> labels(Shape2(test_x.num_instances(), dtrees_.size()));
    for (size_t i = 0; i < dtrees_.size(); ++i)
    {
        auto label_view = labels.template bind<1>(i);
        dtrees_[i].predict(test_x, label_view);
    }

    // Find the majority vote.
    std::vector<size_t> label_counts_vec;
    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        // Count the labels.
        label_counts_vec.resize(0);
        for (size_t k = 0; k < dtrees_.size(); ++k)
        {
            LabelType const label = labels[Shape2(i, k)];
            if (label >= label_counts_vec.size())
                label_counts_vec.resize(label+1);
            ++label_counts_vec[label];
        }

        // Find the label with the maximum count.
        size_t max_count = 0;
        LabelType max_label;
        for (size_t k = 0; k < label_counts_vec.size(); ++k)
        {
            if (label_counts_vec[k] > max_count)
            {
                max_count = label_counts_vec[k];
                max_label = static_cast<LabelType>(k);
            }
        }

        // Write the label in the output array.
        pred_y[i] = max_label;
    }
}






template <typename LABELS>
class PurityTerminationVisitor
{
public:

    PurityTerminationVisitor(LABELS const & data_y)
        : labels_(data_y),
          stop_(true)
    {}

    template <typename TREE>
    void visit(TREE const & tree)
    {
        stop_ = true;
        auto begin = tree.node_sample_begin();
        auto end = tree.node_sample_end();
        if (std::distance(begin, end) < 2)
        {
            return;
        }

        auto const first_label = labels_[*begin];
        ++begin;
        while (begin != end)
        {
            if (labels_[*begin] != first_label)
            {
                stop_ = false;
                return;
            }
            ++begin;
        }
    }

    bool stop() const
    {
        return stop_;
    }

protected:

    LABELS const & labels_;

    /// \brief Whether to stop or continue after the visit.
    bool stop_;
};

class GiniScorer
{
public:
    template <typename COUNTTYPE>
    float operator()(
            std::vector<COUNTTYPE> const & labels_left,
            std::vector<COUNTTYPE> const & labels_prior
    ) const {
        return detail::gini_impurity(labels_left, labels_prior);
    }
};



template <typename FEATURES, typename LABELS, typename SCORER>
class RandomSplitVisitor
{
public:

    typedef FEATURES Features;
    typedef LABELS Labels;
    typedef SCORER Scorer;
    typedef typename Features::value_type FeatureType;
    typedef typename Labels::value_type LabelType;
    typedef UniformIntRandomFunctor<MersenneTwister> Random;

    struct Split
    {
        size_t dim;
        LabelType thresh;
    };

    RandomSplitVisitor(Features const & features, Labels const & labels)
        : features_(features),
          labels_(labels),
          split_({0, 0}),
          feature_indices_(features.num_features()),
          num_feats_(std::ceil(std::sqrt(features.num_features()))),
          rand_(),
          scorer_()
    {
        std::iota(feature_indices_.begin(), feature_indices_.end(), 0);
    }

    template <typename TREE>
    void visit(TREE & tree)
    {
        // Count the labels.
        auto const sample_begin = tree.node_sample_begin();
        auto const sample_end = tree.node_sample_end();
        auto const num_instances = std::distance(sample_begin, sample_end);
        std::fill(tree.labels_prior.begin(), tree.labels_prior.end(), 0);
        for (auto it = sample_begin; it != sample_end; ++it)
        {
            size_t const l = static_cast<size_t>(labels_[*it]);
            if (l > tree.max_label())
            {
                tree.set_max_label(l);
                tree.labels_left.resize(l+1);
                tree.labels_prior.resize(l+1, 0);
            }
            ++tree.labels_prior[l];
        }

        // Get a random subset of the features.
        for (size_t i = 0; i < num_feats_; ++i)
        {
            size_t j = i + (rand_() % (features_.num_features() - i));
            std::swap(feature_indices_[i], feature_indices_[j]);
        }

        // Find the best split.
        split_.dim = 0;
        split_.thresh = 0;
        float best_score = std::numeric_limits<float>::max();
        for (size_t j = 0; j < num_feats_; ++j)
        {
            size_t const feat = feature_indices_[j];

            // Clear the counter of the left child.
            std::fill(tree.labels_left.begin(), tree.labels_left.end(), 0);

            // Iterate over the sorted feature.
            features_.sort(feat, sample_begin, sample_end);
            for (size_t i = 0; i+1 < num_instances; ++i)
            {
                size_t const left_instance = *(sample_begin+i);
                size_t const right_instance = *(sample_begin+i+1);

                // Add the label to the left child.
                size_t const label = static_cast<size_t>(labels_[left_instance]);
                ++tree.labels_left[label];

                // Skip if there is no new split.
                auto const left = features_(left_instance, feat);
                auto const right = features_(right_instance, feat);
                if (left == right)
                    continue;

                // Compute the score.
                float const score = scorer_(tree.labels_left, tree.labels_prior);

                // Update the best score.
                if (score < best_score)
                {
                    best_score = score;
                    split_.dim = feat;
                    split_.thresh = (left+right)/2;
                }
            }
        }
    }

    template <typename ITER>
    ITER apply_split(ITER const & begin, ITER const & end) const
    {
        Features const & features(features_);
        Split const & split(split_);
        return std::partition(begin, end,
                [& features, & split](size_t i) {
                    return features(i, split.dim) < split.thresh;
                }
        );
    }

    Split best_split() const
    {
        return split_;
    }

protected:

    Features const & features_;

    Labels const & labels_;

    Split split_;

    std::vector<size_t> feature_indices_;

    size_t num_feats_;

    Random rand_;

    Scorer scorer_;


};



template <typename FEATURETYPE, typename LABELTYPE>
class ModularDecisionTree
{

public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef BinaryTree Graph;
    typedef Graph::Node Node;
    typedef std::vector<size_t>::iterator iterator;
    typedef detail::IterRange<iterator> IterRange;

    template <typename T>
    using NodeMap = Graph::NodeMap<T>;

    ModularDecisionTree();

    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATIONVISITOR, typename SPLITVISITOR>
    void train(
            FEATURES const & data_x,
            LABELS const & data_y
    );

    iterator node_sample_begin() const
    {
        return node_sample_.begin;
    }

    iterator node_sample_end() const
    {
        return node_sample_.end;
    }

    /// \brief Label counts of the instances in the left child node during training.
    std::vector<size_t> labels_left;

    /// \brief Label counts of all instances during training.
    std::vector<size_t> labels_prior;

    /// \brief Return the max label.
    LabelType max_label() const
    {
        return max_label_;
    }

    /// \brief Set the max label.
    void set_max_label(LabelType const & label)
    {
        max_label_ = label;
    }

protected:

    Graph graph_;

    /// \brief The node queue used in training.
    std::queue<Node> node_queue_;

    /// \brief The instances of each node.
    NodeMap<IterRange> instances_;

    /// \brief The instance sample of the current node during training.
    IterRange node_sample_;

    /// \brief The node labels that were found in training. (Majority label)
    NodeMap<LabelType> node_main_label_;

    /// \brief The node labels that were found in training. (Count of each label)
    NodeMap<std::vector<size_t> > node_labels_; // TODO: Fill this map in training.

    /// \brief The maximum label that was found in training.
    LabelType max_label_;

};

template <typename FEATURETYPE, typename LABELTYPE>
ModularDecisionTree<FEATURETYPE, LABELTYPE>::ModularDecisionTree()
    : labels_left(2),
      labels_prior(2),
      graph_(),
      node_queue_(),
      instances_(),
      node_sample_(),
      node_main_label_(),
      node_labels_(),
      max_label_(0)
{}

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATIONVISITOR, typename SPLITVISITOR>
void ModularDecisionTree<FEATURETYPE, LABELTYPE>::train(
        FEATURES const & data_x,
        LABELS const & data_y
){
    typedef SAMPLER Sampler;
    typedef TERMINATIONVISITOR TermVisitor;
    typedef SPLITVISITOR SplitVisitor;

    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "ModularDecisionTree::train(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "ModularDecisionTree::train(): Wrong label type.");

    size_t num_instances = data_x.num_instances();
    vigra_precondition(num_instances == data_y.num_instances(),
                       "ModularDecisionTree::train(): Input has wrong shape.");

    // Create the bootstrap sample.
    Sampler sampler;
    std::vector<size_t> bootstrap_sample = sampler.bootstrap_sample(num_instances);

    // Create the queue with the nodes to be split and place the root node with the bootstrap samples inside.
    auto const rootnode = graph_.addNode();
    instances_[rootnode] = {bootstrap_sample.begin(), bootstrap_sample.end()};
    node_queue_.push(rootnode);

    // Split the nodes.
    TermVisitor term_visitor(data_y);
    SplitVisitor split_visitor(data_x, data_y);
    while (!node_queue_.empty())
    {
        // Get the next node.
        auto const node = node_queue_.front();
        node_queue_.pop();

        // Draw a random sample of the instances.
        node_sample_ = instances_[node];
        sampler.split_sample(node_sample_.begin, node_sample_.end);

        // Check the termination criterion.
        term_visitor.visit(*this);
        if (term_visitor.stop())
        {
            // Stop splitting and save the node labels.

            // Save the labels of the instances in the node.
            std::vector<size_t> label_count(max_label_+1, 0);
            LabelType best_label = 0;
            size_t best_count = 0;
            for (auto it = node_sample_.begin; it != node_sample_.end; ++it)
            {
                size_t const label = static_cast<size_t>(data_y[*it]);
                size_t count = ++label_count[label];
                if (count > best_count)
                {
                    best_count = count;
                    best_label = label;
                }
            }

            node_main_label_[node] = best_label;
            node_labels_[node] = label_count; // TODO: Maybe use std::move here.
        }
        else
        {
            // Split the node.
            split_visitor.visit(*this);
            auto const split_iter = split_visitor.apply_split(node_sample_.begin, node_sample_.end);

            // Create the new nodes.
            Node const left_node = graph_.addNode();
            Node const right_node = graph_.addNode();
            graph_.addArc(node, left_node);
            graph_.addArc(node, right_node);
            instances_[left_node] = {node_sample_.begin, split_iter};
            instances_[right_node] = {split_iter, node_sample_.end};
            node_queue_.push(left_node);
            node_queue_.push(right_node);
        }
    }
}



template <typename FEATURETYPE, typename LABELTYPE>
class ModularRandomForest
{

public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef ModularDecisionTree<FeatureType, LabelType> Tree;

    ModularRandomForest() = default;
    ModularRandomForest(ModularRandomForest const &) = default;
    ModularRandomForest(ModularRandomForest &&) = default;
    ~ModularRandomForest() = default;
    ModularRandomForest & operator=(ModularRandomForest const &) = default;
    ModularRandomForest & operator=(ModularRandomForest &&) = default;

    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATIONVISITOR, typename SPLITVISITOR>
    void train(
            FEATURES const & data_x,
            LABELS const & data_y,
            size_t num_trees
    );

    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & data_x,
            LABELS & data_y
    ) const;

protected:

    std::vector<Tree> trees_;

};



template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATIONVISITOR, typename SPLITVISITOR>
void ModularRandomForest<FEATURETYPE, LABELTYPE>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "ModularRandomForest::train(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "ModularRandomForest::train(): Wrong label type.");

    trees_.resize(num_trees);
    for (size_t i = 0; i < trees_.size(); ++i)
    {
        std::cout << "training tree " << i << std::endl;
        trees_[i].train<FEATURES, LABELS, SAMPLER, TERMINATIONVISITOR, SPLITVISITOR>(data_x, data_y);
    }
}

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS>
void ModularRandomForest<FEATURETYPE, LABELTYPE>::predict(
        FEATURES const & data_x,
        LABELS & data_y
) const {
    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "ModularRandomForest::predict(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "ModularRandomForest::predict(): Wrong label type.");

    vigra_fail("Not implemented yet.");
}



}

#endif
