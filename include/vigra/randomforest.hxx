#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <vigra/iteratorfacade.hxx>
#include <queue>
#include <map>

#include "dagraph.hxx"

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

    FeatureGetter(MultiArrayView<2, T> const & arr)
        : arr_(arr)
    {}

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

//    template <typename ACCESSOR>
//    reference operator[](ACCESSOR const & x)
//    {
//        return arr_[x];
//    }

//    template <typename ACCESSOR>
//    const_reference operator[](ACCESSOR const & x) const
//    {
//        return arr_[x];
//    }

protected:
    MultiArrayView<2, T> const & arr_;
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

protected:
    MultiArrayView<1, T> const & arr_;
};



/// \brief Simple decision tree class.
template <typename FOREST, typename FEATURES, typename LABELS>
class DecisionTree0
{
public:

    typedef FOREST Forest;
    typedef typename Forest::Node Node;
    typedef FEATURES Features;
    typedef typename Features::value_type FeatureType;
    typedef LABELS Labels;
    typedef typename Labels::value_type LabelType;
    typedef detail::Split<FeatureType> Split;

    template <typename T>
    using PropertyMap = typename Forest::template PropertyMap<T>;

    /// \brief Initialize the tree with the given instance indices.
    /// \param instance_indices: The indices of the instances in the feature matrix.
    DecisionTree0(
            std::vector<size_t> const & instance_indices
    );

    DecisionTree0(DecisionTree0 const &) = default;
    DecisionTree0(DecisionTree0 &&) = default;
    ~DecisionTree0() = default;
    DecisionTree0 & operator=(DecisionTree0 const &) = default;
    DecisionTree0 & operator=(DecisionTree0 &&) = default;

    /// \brief Train the decision tree.
    void train(
            FEATURES const & data_x,
            LABELS const & data_y
    );

    /// \brief Predict new data using the forest.
    void predict(
            FEATURES const & test_x,
            MultiArrayView<1, LabelType> & pred_y
    ) const;

protected:

    /// \brief Split the given node and return the children. If no split was performed, n0 and n1 are set to lemon::INVALID.
    /// \param node: The node that will be split.
    /// \param data_x: The features.
    /// \param data_y: The labels.
    /// \param[out] n0: The first child node.
    /// \param[out] n1: The second child node.
    void split(
            Node const & node,
            FEATURES const & data_x,
            LABELS const & data_y,
            Node & n0,
            Node & n1
    );

    /// \brief The graph structure.
    Forest tree_;

    /// \brief The node labels that were found in training.
    PropertyMap<LabelType> node_labels_;

    /// \brief The split of each node.
    PropertyMap<Split> node_splits_;

    /// \brief Whether the node is a left or a right child.
    PropertyMap<bool> is_left_node;

private:

    typedef std::vector<size_t>::iterator InstanceIterator;
    typedef detail::IterRange<InstanceIterator> Range;

    /// \brief Vector with the instance indices.
    std::vector<size_t> instance_indices_;

    /// \brief The instances of each node (begin and end iterator in the vector instance_indices_).
    PropertyMap<Range> instance_ranges_;

};

template <typename FOREST, typename FEATURES, typename LABELS>
DecisionTree0<FOREST, FEATURES, LABELS>::DecisionTree0(
        const std::vector<size_t> & instance_indices
)   : instance_indices_(instance_indices)
{}

template <typename FOREST, typename FEATURES, typename LABELS>
void DecisionTree0<FOREST, FEATURES, LABELS>::train(
        FEATURES const & data_x,
        LABELS const & data_y
){
    // Create the queue with the nodes to be split and place the root node with all instances inside.
    std::queue<Node> node_queue;
    auto const rootnode = tree_.addNode();
    instance_ranges_[rootnode] = {instance_indices_.begin(), instance_indices_.end()};
    node_queue.push(rootnode);

    // Split the nodes.
    while (!node_queue.empty())
    {
        auto const node = node_queue.front();
        node_queue.pop();

        Node n0, n1;
        split(node, data_x, data_y, n0, n1);
        if (tree_.valid(n0))
            node_queue.push(n0);
        if (tree_.valid(n1))
            node_queue.push(n1);
    }
}

template <typename FOREST, typename FEATURES, typename LABELS>
void DecisionTree0<FOREST, FEATURES, LABELS>::predict(
        FEATURES const & test_x,
        MultiArrayView<1, LabelType> & pred_y
) const {

    typedef typename Forest::RootNodeIt RootNodeIt;
    typedef typename Forest::ChildIt ChildIt;

    Node rootnode = RootNodeIt(tree_);
    vigra_assert(tree_.valid(rootnode), "DecisionTree0::predict(): The graph has no root node.");

    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        auto const feats = test_x.instance_features(i);
        Node node = rootnode;

        while (!tree_.isLeafNode(node))
        {
             auto const & s = node_splits_.at(node);
             bool go_left = feats[s.feature_index] < s.thresh;

             for (ChildIt it(tree_, node); it != lemon::INVALID; ++it)
             {
                 auto const tmp = Node(it);
                 if ((go_left && is_left_node.at(tmp)) || (!go_left && !is_left_node.at(tmp)))
                 {
                     node = tmp;
                     break;
                 }
             }
        }
        pred_y[i] = node_labels_.at(node);
    }
}

template <typename FOREST, typename FEATURES, typename LABELS>
void DecisionTree0<FOREST, FEATURES, LABELS>::split(
        Node const & node,
        FEATURES const & data_x,
        LABELS const & data_y,
        Node & n0,
        Node & n1
){
    auto const inst_begin = instance_ranges_[node].begin;
    auto const inst_end = instance_ranges_[node].end;
    auto const num_instances = std::distance(inst_begin, inst_end);

//    // TODO: Remove output.
//    std::cout << "splitting node " << node << " with "
//              << num_instances << " instances" << std::endl;

    // Check whether the given node is pure.
    {
        auto is_pure = true;
        LabelType first_label = 0;
        if (num_instances > 1)
        {
            auto it(inst_begin);
            first_label = data_y[*it];
            for (++it; it != inst_end; ++it)
            {
                if (data_y[*it] != first_label)
                {
                    is_pure = false;
                    break;
                }
            }
        }
        if (is_pure)
        {
            node_labels_[node] = first_label;
            n0 = lemon::INVALID;
            n1 = lemon::INVALID;
            return;
        }
    }

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
    // Note: A map would be more natural, but a vector offers faster access.
    std::vector<size_t> label_priors;
    for (InstanceIterator it(inst_begin); it != inst_end; ++it)
    {
        size_t const l = static_cast<size_t>(data_y[*it]);
        if (l >= label_priors.size())
            label_priors.resize(l+1);
        ++label_priors[l];
    }

    // Find the best split.
    size_t best_feat = 0;
    FeatureType best_split = 0;
    float best_gini = std::numeric_limits<float>::max();
    for (auto const & feat : feat_indices)
    {
        // Sort the instances according to the current feature.
        auto const features_multiarray = data_x.get_features(feat);
        std::sort(inst_begin, inst_end,
                [& features_multiarray](size_t a, size_t b){
                    return features_multiarray[a] < features_multiarray[b];
                }
        );

        // Sort the features likewise.
        std::vector<FeatureType> features_vec;
        features_vec.reserve(features_multiarray.size());
        for (InstanceIterator it(inst_begin); it != inst_end; ++it)
            features_vec.push_back(features_multiarray[*it]);

        // Compute the splits.
        // TODO: Use split iterator.
        std::vector<FeatureType> splits;
        for (InstanceIterator it0(inst_begin), it1(inst_begin+1); it0 != inst_end && it1 != inst_end; ++it0, ++it1)
        {
            auto const & f0 = features_multiarray[*it0];
            auto const & f1 = features_multiarray[*it1];
            if (f0 != f1)
                splits.push_back((f0+f1)/2);
        }

        // This vector keeps track of the labels of the instances that are assigned to the left child.
        // Note: A map would be more natural, but a vector offers faster access.
        std::vector<size_t> labels_left;

        // Compute the gini impurity of each split.
        size_t first_right_index = 0; // index of the first instance that is assigned to the right child
        for (auto const & s : splits)
        {
            // Add the new labels to the left child.
            do
            {
                size_t const new_label = static_cast<size_t>(data_y[*(inst_begin+first_right_index)]);
                if (new_label >= labels_left.size())
                    labels_left.resize(new_label+1);
                ++labels_left[new_label];
                ++first_right_index;
            }
            while (features_vec[first_right_index] < s);

            // Compute the gini.
            float const gini = detail::gini_impurity(labels_left, label_priors);
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

//    // TODO: Remove output.
//    std::cout << "divided into " << std::distance(begin, split_iter) << " and " << std::distance(split_iter, end) << std::endl;
}



/// \brief Random forest class.
/// \note FEATURES/LABELS must implement the operator[] that gets an instance index and returns the features/labels of that instance.
template <typename FEATURES, typename LABELS>
class RandomForest0
{
public:

    typedef vigra::Forest1<vigra::DAGraph0> Forest;
    typedef Forest::Node Node;
    typedef FEATURES Features;
    typedef typename Features::value_type FeatureType;
    typedef LABELS Labels;
    typedef typename Labels::value_type LabelType;

    template <typename VALUE_TYPE>
    using PropertyMap = Forest::template PropertyMap<VALUE_TYPE>;

    RandomForest0() = default;
    RandomForest0(RandomForest0 const &) = default;
    RandomForest0(RandomForest0 &&) = default;
    ~RandomForest0() = default;
    RandomForest0 & operator=(RandomForest0 const &) = default;
    RandomForest0 & operator=(RandomForest0 &&) = default;

    /// \brief Train the random forest.
    void train(
            FEATURES const & train_x,
            LABELS const & train_y,
            size_t num_trees
    );

    /// \brief Predict new data using the forest.
    void predict(
            FEATURES const & test_x,
            MultiArrayView<1, LabelType> & pred_y
    ) const;

protected:

    /// \brief The trees of the forest.
    std::vector<DecisionTree0<Forest, Features, Labels> > dtrees_;

};

template <typename FEATURES, typename LABELS>
void RandomForest0<FEATURES, LABELS>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    dtrees_.reserve(num_trees);

    for (size_t i = 0; i < num_trees; ++i)
    {
        // TODO: Remove output.
        std::cout << "training tree " << i << std::endl;

        // Draw the bootstrap indices.
        std::vector<size_t> index_vector;
        index_vector.reserve(data_x.num_instances());
        for (size_t k = 0; k < data_x.num_instances(); ++k)
            index_vector.push_back(k);
        std::vector<size_t> instance_indices;
        instance_indices.reserve(data_x.num_instances());
        detail::sample_with_replacement(
                    data_x.num_instances(),
                    index_vector.begin(),
                    index_vector.end(),
                    std::back_inserter(instance_indices)
        );

        // Create the tree and train it.
        dtrees_.push_back({instance_indices});
        dtrees_.back().train(data_x, data_y);
    }
}

template <typename FEATURES, typename LABELS>
void RandomForest0<FEATURES, LABELS>::predict(
        FEATURES const & test_x,
        MultiArrayView<1, LabelType> & pred_y
) const {

    // Let each tree predict all instances.
    MultiArray<2, LabelType> labels(Shape2(test_x.num_instances(), dtrees_.size()));
    for (size_t i = 0; i < dtrees_.size(); ++i)
    {
        auto label_view = labels.template bind<1>(i);
        dtrees_[i].predict(test_x, label_view);
    }

    // Find the majority vote.
    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        // Count the labels.
        std::vector<size_t> label_counts_vec;
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



}

#endif
