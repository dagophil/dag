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

//    /// \brief Compute the gini impurity.
//    /// \param labels_left: Label counts of the left child.
//    /// \param label_priors: Total label count.
//    template <typename COUNTTYPE>
//    float gini_impurity(
//            std::vector<COUNTTYPE> const & labels_left,
//            std::vector<COUNTTYPE> const & label_priors
//    ){
//        typedef COUNTTYPE CountType;

//        CountType const n_total = std::accumulate(
//                    label_priors.begin(),
//                    label_priors.end(),
//                    0
//        );
//        CountType const n_left = std::accumulate(
//                    labels_left.begin(),
//                    labels_left.end(),
//                    0
//        );
//        CountType const n_right = n_total - n_left;

//        float gini_left = 1;
//        float gini_right = 1;
//        for (size_t i = 0; i < labels_left.size(); ++i)
//        {
//            float const p_left = labels_left[i] / static_cast<float>(n_left);
//            float const p_right = (label_priors[i] - labels_left[i]) / static_cast<float>(n_right);
//            gini_left -= (p_left*p_left);
//            gini_right -= (p_right*p_right);
//        }
//        return n_left*gini_left + n_right*gini_right;
//    }

} // namespace detail

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

    reference operator()(size_t i, size_t j)
    {
        return arr_(i, j);
    }

    const_reference operator()(size_t i, size_t j) const
    {
        return arr_(i, j);
    }

    template <typename ITER>
    void sort(size_t feat, ITER begin, ITER end) const
    {
        auto const & arr = arr_;
        std::sort(begin, end,
                [& arr, & feat](size_t i, size_t j)
                {
                    return arr(i, feat) < arr(j, feat);
                }
        );
    }

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



class GiniScorer
{
public:

    template <typename LABELS, typename ITER>
    GiniScorer(LABELS const & labels, ITER begin, ITER end)
        : labels_prior_(0),
          labels_left_(0),
          n_total_(std::distance(begin, end)),
          n_left_(0)
    {
        for (auto it = begin; it != end; ++it)
        {
            size_t label = labels[*it];
            if (label >= labels_prior_.size())
                labels_prior_.resize(label+1);
            ++labels_prior_[label];
        }
        labels_left_.resize(labels_prior_.size());
    }

    void add_left(size_t label)
    {
        if (labels_left_.size() <= label)
            labels_left_.resize(label+1);
        ++labels_left_[label];
        ++n_left_;
    }

    void clear_left()
    {
        std::fill(labels_left_.begin(), labels_left_.end(), 0);
        n_left_ = 0;
    }

    float operator()() const {
        float n_left = static_cast<float>(n_left_);
        float n_right = static_cast<float>(n_total_ - n_left_);
        float gini_left = 1;
        float gini_right = 1;
        for (size_t i = 0; i < labels_left_.size(); ++i)
        {
            float const p_left = labels_left_[i] / n_left;
            float const p_right = (labels_prior_[i] - labels_left_[i]) / n_right;
            gini_left -= (p_left*p_left);
            gini_right -= (p_right*p_right);
        }
        return n_left*gini_left + n_right*gini_right;
    }

protected:

    std::vector<size_t> labels_prior_;
    std::vector<size_t> labels_left_;
    size_t const n_total_;
    size_t n_left_;
};



template <typename RAND, typename SCORER>
class RandomSplit
{
public:

    RandomSplit(RAND const & rand)
        : rand_(rand)
    {}

    template <typename ITER, typename FEATURES, typename LABELS>
    bool split(
            ITER const inst_begin,
            ITER const inst_end,
            FEATURES const & features,
            LABELS const & labels,
            size_t & best_feat,
            typename FEATURES::value_type & best_split,
            ITER & split_iter
    ) const {
        auto const num_instances = std::distance(inst_begin, inst_end);

        // Get a random subset of the features.
        size_t const num_feats = std::ceil(std::sqrt(features.num_features()));
        std::vector<size_t> all_feat_indices(features.num_features());
        std::iota(all_feat_indices.begin(), all_feat_indices.end(), 0);
        for (size_t i = 0; i < num_feats; ++i)
        {
            size_t j = rand_() % (features.num_features()-i);
            std::swap(all_feat_indices[i], all_feat_indices[j]);
        }

        // Initialize the scorer with the labels.
        SCORER scorer(labels, inst_begin, inst_end);

        // On small sets, it might happen that all features on the random
        // feature subset are equal. In that case, no split was considered
        // at all and the function returns false.
        bool split_found = false;

        // Find the best split.
        float best_score = std::numeric_limits<float>::max();
        for (size_t k = 0; k < num_feats; ++k)
        {
            auto const feat = all_feat_indices[k];

            // Sort the instances according to the current feature.
            features.sort(feat, inst_begin, inst_end);

            // Compute the score of each split.
            scorer.clear_left();
            for (size_t i = 0; i+1 < num_instances; ++i)
            {
                // Compute the split.
                size_t const left_instance = *(inst_begin+i);
                size_t const right_instance = *(inst_begin+i+1);

                // Add the label to the left child.
                size_t const label = static_cast<size_t>(labels[left_instance]);
                scorer.add_left(label);

                // Skip if there is no new split.
                auto const left = features(left_instance, feat);
                auto const right = features(right_instance, feat);
                if (left == right)
                    continue;

                // Update the best score.
                split_found = true;
                float const score = scorer();
                if (score < best_score)
                {
                    best_score = score;
                    best_split = (left+right)/2;
                    best_feat = feat;
                }
            }
        }

        if (!split_found)
            return false;

        // Separate the data according to the best split.
        auto const best_features = features.get_features(best_feat);
        split_iter = std::partition(inst_begin, inst_end,
                [& best_features, & best_split](size_t instance_index){
                    return best_features[instance_index] < best_split;
                }
        );
        return true;
    }

protected:

    RAND const & rand_;

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
    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
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

private:

    typedef detail::IterRange<std::vector<size_t>::iterator > Range;

    /// \brief The instances of each node (begin and end iterator in the vector instance_indices_).
    NodeMap<Range> instance_ranges_;

};

template <typename FEATURETYPE, typename LABELTYPE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
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

    // Initialize the split functor with a random engine.
    UniformIntRandomFunctor<MersenneTwister> rand;
    SPLITFUNCTOR functor(rand);

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
        bool split_found = false;
        if (do_split)
        {
            // Split the node.
            size_t best_feat;
            FeatureType best_split;
            std::vector<size_t>::iterator split_iter;
            split_found = functor.split(instances.begin, instances.end, features, labels, best_feat, best_split, split_iter);
            if (split_found)
            {
                // Add the child nodes to the graph.
                Node n0 = tree_.addNode();
                Node n1 = tree_.addNode();
                tree_.addArc(node, n0);
                tree_.addArc(node, n1);
                instance_ranges_[n0] = {instances.begin, split_iter};
                instance_ranges_[n1] = {split_iter, instances.end};
                node_splits_[node] = {best_feat, best_split};
                node_queue.push(n0);
                node_queue.push(n1);
            }
        }

        if (!do_split || !split_found)
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
    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename RANDOMSPLIT>
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
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
void RandomForest0<FEATURETYPE, LABELTYPE>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    static_assert(std::is_same<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::train(): Wrong feature type.");
    static_assert(std::is_same<typename LABELS::value_type, LabelType>(),
                  "RandomForest0::train(): Wrong label type.");

    dtrees_.resize(num_trees);
    for (size_t i = 0; i < num_trees; ++i)
    {
        // TODO: Remove output.
        std::cout << "training tree " << i << std::endl;
        dtrees_[i].train<FEATURES, LABELS, SAMPLER, TERMINATION, SPLITFUNCTOR>(data_x, data_y);
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



}

#endif
