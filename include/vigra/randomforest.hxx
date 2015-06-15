#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <map>
#include <set>
#include <type_traits>
#include <thread>
#include <stack>

//#include "dagraph.hxx"
#include "jungle.hxx"
#include "feature_getter.hxx"
#include "svm.hxx"


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
            size_t k = rand(num_instances);
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
            size_t k = rand(num_instances-i);
            *out = values[k];
            ++out;
            values[k] = values[num_instances-i-1];
        }
    }

//    /// \brief Compute the gini impurity.
//    /// \param labels_left: Label counts of the left child.
//    /// \param label_priors: Total label count.
//    template <typename COUNTTYPE>
//    double gini_impurity(
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

//        double gini_left = 1;
//        double gini_right = 1;
//        for (size_t i = 0; i < labels_left.size(); ++i)
//        {
//            double const p_left = labels_left[i] / static_cast<double>(n_left);
//            double const p_right = (label_priors[i] - labels_left[i]) / static_cast<double>(n_right);
//            gini_left -= (p_left*p_left);
//            gini_right -= (p_right*p_right);
//        }
//        return n_left*gini_left + n_right*gini_right;
//    }

} // namespace detail



class BootstrapSampler
{

public:

    typedef std::vector<size_t>::iterator iterator;
    typedef std::vector<size_t>::const_iterator const_iterator;

    /// \brief Create a bootstrap sample.
    template <typename RANDENGINE>
    std::vector<size_t> bootstrap_sample(size_t num_instances, RANDENGINE const & randengine) const
    {
        UniformIntRandomFunctor<RANDENGINE> rand(randengine);
        std::vector<size_t> v(num_instances);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = rand(num_instances);
        return v;
    }

    /// \brief Return all of the given instances (hence do nothing).
    template <typename ITER>
    void split_sample(ITER & begin, ITER & end) const
    {
    }

protected:

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

        first_label = labels(*begin);
        ++begin;
        while (begin != end)
        {
            if (labels(*begin) != first_label)
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
    GiniScorer(LABELS const & labels, size_t const num_labels, ITER begin, ITER end)
        : labels_prior_(num_labels),
          labels_left_(num_labels),
          n_total_(std::distance(begin, end)),
          n_left_(0)
    {
        for (auto it = begin; it != end; ++it)
        {
            size_t label = labels(*it);
            if (label >= labels_prior_.size())
                vigra_fail("GiniScorer(): Max label is larger than expected.");
            ++labels_prior_[label];
        }
    }

    void add_left(size_t label)
    {
        if (label >= labels_left_.size())
            vigra_fail("GiniScorer::add_left(): Label is larger than expected.");
        ++labels_left_[label];
        ++n_left_;
    }

    void clear_left()
    {
        std::fill(labels_left_.begin(), labels_left_.end(), 0);
        n_left_ = 0;
    }

    double operator()() const {
        double const n_left = static_cast<double>(n_left_);
        double const n_right = static_cast<double>(n_total_ - n_left_);
        double gini_left = 1;
        double gini_right = 1;
        for (size_t i = 0; i < labels_left_.size(); ++i)
        {
            double const p_left = labels_left_[i] / n_left;
            double const p_right = (labels_prior_[i] - labels_left_[i]) / n_right;
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



template <typename SCORER>
class RandomSplit
{
public:

    /// \todo Add doc comment.
    template <typename ITER, typename FEATURES, typename LABELS, typename RANDENGINE>
    bool split(
            ITER const inst_begin,
            ITER const inst_end,
            FEATURES const & features,
            LABELS const & labels,
            size_t const num_labels,
            RANDENGINE const & randengine,
            size_t & best_feat,
            typename FEATURES::value_type & best_split,
            ITER & split_iter
    ) const {
        auto const num_instances = std::distance(inst_begin, inst_end);
        auto const num_features = features.shape()[1];

        // Get a random subset of the features.
        UniformIntRandomFunctor<RANDENGINE> rand(randengine);
        size_t const num_feats = std::ceil(std::sqrt(num_features));
        std::vector<size_t> all_feat_indices(num_features);
        std::iota(all_feat_indices.begin(), all_feat_indices.end(), 0);
        for (size_t i = 0; i < num_feats; ++i)
        {
            size_t j = i + (rand(num_features-i));
            std::swap(all_feat_indices[i], all_feat_indices[j]);
        }

        // Initialize the scorer with the labels.
        SCORER scorer(labels, num_labels, inst_begin, inst_end);

        // On small sets, it might happen that all features on the random
        // feature subset are equal. In that case, no split was considered
        // at all and the function returns false.
        bool split_found = false;

        // Find the best split.
        double best_score = std::numeric_limits<double>::max();
        for (size_t k = 0; k < num_feats; ++k)
        {
            auto const feat = all_feat_indices[k];

            // Sort the instances according to the current feature.
            std::sort(inst_begin, inst_end,
                    [& features, & feat](size_t i, size_t j)
                    {
                        return features(i, feat) < features(j, feat);
                    }
            );

            // Compute the score of each split.
            scorer.clear_left();
            for (size_t i = 0; i+1 < num_instances; ++i)
            {
                // Compute the split.
                size_t const left_instance = inst_begin[i];
                size_t const right_instance = inst_begin[i+1];

                // Add the label to the left child.
                size_t const label = static_cast<size_t>(labels(left_instance));
                scorer.add_left(label);

                // Skip if there is no new split.
                auto const left = features(left_instance, feat);
                auto const right = features(right_instance, feat);
                if (left == right)
                    continue;

                // Update the best score.
                split_found = true;
                double const score = scorer();
                if (score < best_score)
                {
                    best_score = score;
                    best_split = 0.5*(left+right);
                    best_feat = feat;
                }
            }
        }

        if (!split_found)
            return false;

        // Separate the data according to the best split.
        split_iter = std::partition(inst_begin, inst_end,
                [& features, & best_feat, & best_split](size_t instance_index)
                {
                    return features(instance_index, best_feat) < best_split;
                }
        );
        return true;
    }
};



/// \brief Simple decision tree class.
template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE = MersenneTwister>
class DecisionTree0
{
public:

    typedef BinaryTree Graph;
    typedef typename Graph::Node Node;
    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef detail::Split<FeatureType> Split;
    typedef RANDENGINE Randengine;

    template <typename T>
    using NodeMap = typename Graph::template NodeMap<T>;

    DecisionTree0(size_t const seed)
        : tree_(),
          node_main_label_(),
          node_splits_(),
          num_labels_(0),
          randengine_(seed),
          instance_ranges_()
    {}

    DecisionTree0(DecisionTree0 const &) = default;
    DecisionTree0(DecisionTree0 &&) = default;
    ~DecisionTree0() = default;
    DecisionTree0 & operator=(DecisionTree0 const &) = default;
    DecisionTree0 & operator=(DecisionTree0 &&) = default;

    /// \brief Train the decision tree.
    ///
    /// \note Before calling train, you must call set_num_labels with a value larger than the maximum value in data_y.
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

    /// \brief Return the number of labels.
    void get_num_labels() const
    {
        return num_labels_;
    }

    /// \brief Set the number of labels.
    void set_num_labels(size_t num_labels)
    {
        num_labels_ = num_labels;
    }

    /// \brief Return the number of leaves.
    size_t num_leaves() const
    {
        return tree_.numLeaves();
    }

    /// \brief Return the label of the leaf with the given index.
    LabelType leaf_main_label(size_t leaf_index) const
    {
        return node_main_label_.at(tree_.getLeafNode(leaf_index));
    }

    /// \brief Return the graph structure.
    Graph const & get_graph() const
    {
        return tree_;
    }

    /// \brief Return the node splits.
    NodeMap<Split> const & node_splits() const
    {
        return node_splits_;
    }

    /// \brief Return the node labels.
    NodeMap<LabelType> const & node_main_label() const
    {
        return node_main_label_;
    }

    /// \brief Return the class probabilities.
    NodeMap<std::vector<double> > const & label_probs() const
    {
        return label_probs_;
    }

    /// \brief Return the node ids of the leaves that contain the given instances.
    template <typename FEATURES>
    void leaf_ids(
            FEATURES const & features,
            MultiArrayView<1, size_t> & indices
    ) const;

protected:

    /// \brief The graph structure.
    Graph tree_;

    /// \brief The node labels that were found in training.
    NodeMap<LabelType> node_main_label_;

    /// \brief Node map with the probabilities of the classes in each leaf.
    NodeMap<std::vector<double> > label_probs_;

    /// \brief The number of instances in each leaf.
    NodeMap<size_t> instance_count_;

    /// \brief The split of each node.
    NodeMap<Split> node_splits_;

    /// \brief The number of distinct labels.
    size_t num_labels_;

    /// \brief The random engine.
    Randengine randengine_;

private:

    typedef detail::IterRange<std::vector<size_t>::iterator > Range;

    /// \brief The instances of each node (begin and end iterator in the vector instance_indices_).
    NodeMap<Range> instance_ranges_;

};

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
void DecisionTree0<FEATURETYPE, LABELTYPE, RANDENGINE>::train(
        FEATURES const & features,
        LABELS const & labels
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "DecisionTree0::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "DecisionTree0::train(): Wrong label type.");

    vigra_precondition(num_labels_ > 0, "DecisionTree::train(): The number of distinct labels must be set before training.");

    // Create the bootstrap indices.
    SAMPLER sampler;
    std::vector<size_t> instance_indices = sampler.bootstrap_sample(labels.size(), randengine_);

    // Create the queue with the nodes to be split and place the root node with all instances inside.
    std::stack<Node> node_stack;
    auto const rootnode = tree_.addNode();
    instance_ranges_[rootnode] = {instance_indices.begin(), instance_indices.end()};
    node_stack.push(rootnode);

    // Initialize the split functor with a random engine.
    SPLITFUNCTOR functor;

    // Split the nodes.
    while (!node_stack.empty())
    {
        auto const node = node_stack.top();
        node_stack.pop();

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
            split_found = functor.split(instances.begin, instances.end, features, labels, num_labels_, randengine_, best_feat, best_split, split_iter);
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
                node_stack.push(n0);
                node_stack.push(n1);
            }
        }

        if (!do_split || !split_found)
        {
            // Make the node terminal.

            // Count the instances and get the probabilities of each class.
            size_t count = 0;
            std::vector<double> probs(num_labels_);
            for (auto it = instances.begin; it != instances.end; ++it)
            {
                ++count;
                ++probs[labels(*it)];
            }
            for (size_t i = 0; i < probs.size(); ++i)
            {
                probs[i] /= count;
            }

            // Save the data in the node maps.
            label_probs_.emplace(node, probs);
            instance_count_[node] = count;
            node_main_label_[node] = first_label;
        }
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS>
void DecisionTree0<FEATURETYPE, LABELTYPE, RANDENGINE>::predict(
        FEATURES const & test_x,
        LABELS & pred_y
) const {
    // The features must be convertible to the internal feature type, so we can put them into the tree.
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "DecisionTree0::predict(): Wrong feature type.");

    // The internal label type must be convertible to the given type, so we can put them into the output.
    static_assert(std::is_convertible<LabelType, typename LABELS::value_type>(),
                  "DecisionTree0::predict(): Wrong label type.");

    Node const rootnode = tree_.getRoot();
    vigra_assert(tree_.valid(rootnode), "DecisionTree0::predict(): The graph has no root node.");

    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        auto const feats = test_x.instance_features(i);
        Node node = rootnode;

        while (tree_.outDegree(node) > 0)
        {
             auto const & s = node_splits_.at(node);
             if (feats(s.feature_index) < s.thresh)
             {
                 node = tree_.getChild(node, 0);
             }
             else
             {
                 node = tree_.getChild(node, 1);
             }
        }
        pred_y(i) = node_main_label_.at(node);
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES>
void DecisionTree0<FEATURETYPE, LABELTYPE, RANDENGINE>::leaf_ids(
        FEATURES const & features,
        MultiArrayView<1, size_t> & indices
) const {
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "DecisionTree0::leaf_indices(): Wrong feature type.");

    size_t const num_instances = features.shape()[0];

    vigra_precondition(num_instances == indices.size(),
                       "DecisionTree0::leaf_indices(): Shape mismatch.");

    Node const rootnode = tree_.getRoot();
    vigra_assert(tree_.valid(rootnode), "DecisionTree0::leaf_indices(): The graph has no root node.");

    for (size_t i = 0; i < num_instances; ++i)
    {
        Node node = rootnode;

        while (tree_.outDegree(node) > 0)
        {
            auto const & s = node_splits_.at(node);
            if (features(i, s.feature_index) < s.thresh)
            {
                node = tree_.getChild(node, 0);
            }
            else
            {
                node = tree_.getChild(node, 1);
            }
        }
        indices(i) = node.id();
    }
}



/// \brief Random forest class.
template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE = MersenneTwister>
class RandomForest0
{
public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef DecisionTree0<FeatureType, size_t, RANDENGINE> Tree;
    typedef typename Tree::Node TreeNode;

    RandomForest0(RANDENGINE const & randengine = RANDENGINE::global())
        : randengine_(randengine)
    {}

    RandomForest0(RandomForest0 const &) = default;
    RandomForest0(RandomForest0 &&) = default;
    ~RandomForest0() = default;
    RandomForest0 & operator=(RandomForest0 const &) = default;
    RandomForest0 & operator=(RandomForest0 &&) = default;

    /// \brief Train the random forest.
    template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
    void train(
            FEATURES const & train_x,
            LABELS const & train_y,
            size_t num_trees,
            int num_threads = -1
    );

    /// \brief Predict new data using the forest.
    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & test_x,
            LABELS & pred_y
    ) const;

    /// \brief Return the tree vector.
    std::vector<Tree> & trees()
    {
        return dtrees_;
    }


    /// \brief Return the tree vector.
    std::vector<Tree> const & trees() const
    {
        return dtrees_;
    }

    /// \brief Return the number of trees.
    size_t num_trees() const
    {
        return dtrees_.size();
    }

    /// \brief Return the number of classes.
    size_t num_classes() const
    {
        return distinct_labels_.size();
    }

    /// \brief Return the graph vector.
    std::vector<typename Tree::Graph> get_graph() const
    {
        std::vector<typename Tree::Graph> graph;
        for (Tree const & tree : dtrees_)
        {
            graph.push_back(tree.get_graph());
        }
        return graph;
    }

    /// \brief For each tree return the node ids of the leaves that contain the given instances.
    template <typename FEATURES>
    void leaf_ids(
            FEATURES const & features,
            MultiArrayView<2, size_t> & indices
    ) const;

    /// \brief Transform the given external labels to the labels that are used internally.
    template <typename LABELS>
    void transform_external_labels(
            LABELS const & labels_in,
            MultiArrayView<1, size_t> & labels_out
    ) const;

protected:

    /// \brief The trees of the forest.
    std::vector<Tree> dtrees_;

    /// \brief The distinct labels that were found in training.
    std::vector<LabelType> distinct_labels_;

    /// \brief The random engine.
    RANDENGINE const & randengine_;

};

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS, typename SAMPLER, typename TERMINATION, typename SPLITFUNCTOR>
void RandomForest0<FEATURETYPE, LABELTYPE, RANDENGINE>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t const num_trees,
        int num_threads
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "RandomForest0::train(): Wrong label type.");

    vigra_precondition(num_threads == -1 || num_threads > 0,
                       "RandomForest0::train(): n_threads must be -1 or greater than zero.");

    // Find the distinct labels.
    std::set<LabelType> dlabels(data_y.begin(), data_y.end());
    distinct_labels_.resize(dlabels.size());
    std::copy(dlabels.begin(), dlabels.end(), distinct_labels_.begin());

    // Translate the labels to the label ids.
    MultiArray<1, size_t> data_y_id_arr(data_y.shape());
    transform_external_labels(data_y, data_y_id_arr);
    LabelGetter<size_t> data_y_id(data_y_id_arr);

    // Create a named lambda to train a single tree with index i.
    auto train_tree = [this, & data_x, & data_y_id](size_t i) {
        // TODO: Remove output.
        std::cout << "training tree " << i << std::endl;
        dtrees_[i].set_num_labels(distinct_labels_.size());
        dtrees_[i].train<FEATURES, LabelGetter<size_t>, SAMPLER, TERMINATION, SPLITFUNCTOR>(data_x, data_y_id);
    };

    // Create the seeds for the trees. Make sure that they are all different.
    UniformIntRandomFunctor<RANDENGINE> rand(randengine_);
    std::set<size_t> seeds;
    while (seeds.size() < num_trees)
    {
        seeds.insert(rand());
    }
    vigra_assert(seeds.size() == num_trees, "RandomForest0::train(): Somehow we got more seeds than trees.");

    // Initialize the trees with the seeds.
    dtrees_.reserve(num_trees);
    for (auto it = seeds.begin(); it != seeds.end(); ++it)
    {
        dtrees_.push_back(Tree(*it));
    }

    // Train each tree.
    if (num_threads == -1)
        num_threads = std::thread::hardware_concurrency(); // might return 0 if the value is not computable
    if (num_threads <= 1)
    {
        // Single thread.
        for (size_t i = 0; i < num_trees; ++i)
        {
            train_tree(i);
        }
    }
    else
    {
        // Create one worker per thread and let them train the trees.
        std::vector<std::thread> workers;
        for (size_t k = 0; k < num_threads; ++k)
        {
            workers.push_back(std::thread(
                    [this, & train_tree, & num_trees, & num_threads](size_t k)
                    {
                        for (size_t i = k; i < num_trees; i+=num_threads)
                        {
                            train_tree(i);
                        }
                    },
                    k
            ));
        }
        for (auto & t : workers)
        {
            t.join();
        }
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS>
void RandomForest0<FEATURETYPE, LABELTYPE, RANDENGINE>::predict(
        FEATURES const & test_x,
        LABELS & pred_y
) const {
    // The features must be convertible to feature type, so we can put them into the forest.
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::predict(): Wrong feature type.");

    // The internal label type must be convertible to the labels, so we can put them into the output.
    static_assert(std::is_convertible<LabelType, typename LABELS::value_type>(),
                  "RandomForest0::predict(): Wrong label type.");

    // Let each tree predict all instances.
    MultiArray<2, size_t> labels(Shape2(test_x.num_instances(), dtrees_.size()));
    for (size_t i = 0; i < dtrees_.size(); ++i)
    {
        auto label_view = labels.template bind<1>(i);
        dtrees_[i].predict(test_x, label_view);
    }

    // Find the majority vote.
    std::vector<size_t> label_counts_vec(distinct_labels_.size());
    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        // Count the labels.
        std::fill(label_counts_vec.begin(), label_counts_vec.end(), 0);
        for (size_t k = 0; k < dtrees_.size(); ++k)
        {
            size_t const label = labels(i, k);
            if (label >= label_counts_vec.size())
                vigra_fail("Prediction of a label that did not exist in training.");
            ++label_counts_vec[label];
        }

        // Find the label with the maximum count.
        size_t max_count = 0;
        size_t max_label;
        for (size_t k = 0; k < label_counts_vec.size(); ++k)
        {
            if (label_counts_vec[k] > max_count)
            {
                max_count = label_counts_vec[k];
                max_label = k;
            }
        }

        // Write the label in the output array.
        pred_y(i) = distinct_labels_[max_label];
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES>
void RandomForest0<FEATURETYPE, LABELTYPE, RANDENGINE>::leaf_ids(
        FEATURES const & features,
        MultiArrayView<2, size_t> & indices
) const {
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "RandomForest0::leaf_indices(): Wrong feature type.");

    vigra_precondition(indices.shape() == Shape2(features.shape()[0], dtrees_.size()),
                       "RandomForest0::leaf_indices(): Shape mismatch.");

    for (size_t i = 0; i < dtrees_.size(); ++i)
    {
        auto sub = indices.bind<1>(i);
        dtrees_[i].leaf_ids(features, sub);
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename LABELS>
void RandomForest0<FEATURETYPE, LABELTYPE, RANDENGINE>::transform_external_labels(
        LABELS const & labels_in,
        MultiArrayView<1, size_t> & labels_out
) const {
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "RandomForest0::transform_external_labels(): Wrong label type.");

    vigra_precondition(labels_in.size() == labels_out.size(),
                       "RandomForest0::transform_external_labels(): Shape mismatch.");

    std::map<FeatureType, size_t> label_id;
    for (size_t i = 0; i < distinct_labels_.size(); ++i)
    {
        label_id[distinct_labels_[i]] = i;
    }
    for (size_t i = 0; i < labels_in.size(); ++i)
    {
        labels_out(i) = label_id[labels_in(i)];
    }
}



template <typename RANDOMFOREST>
class GloballyRefinedRandomForest
{
public:

    typedef RANDOMFOREST RandomForest;
    typedef typename RandomForest::FeatureType FeatureType;
    typedef typename RandomForest::LabelType LabelType;
    typedef typename RandomForest::Tree Tree;
    typedef typename Tree::Graph TreeGraph;
    typedef typename Tree::Node TreeNode;

    typedef ConstForestAdaptor<TreeGraph> ForestAdaptor;
    typedef typename ForestAdaptor::Node Node;

    typedef TwoClassSVM<UInt8, size_t> SVM;

    template <typename T>
    using NodeMap = typename ForestAdaptor::template NodeMap<T>;

    template <typename T>
    using TreeNodeMap = typename Tree::template NodeMap<T>;

    GloballyRefinedRandomForest(RANDOMFOREST const & rf)
        : rf_(rf)
    {}

    template <typename FEATURES, typename LABELS>
    void train(
            FEATURES const & features,
            LABELS const & labels
    );

    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & features,
            LABELS & pred_y
    ) const;

protected:

    RandomForest const & rf_;

    ForestAdaptor rf_adaptor_;

    std::vector<TreeNodeMap<double> > svm_weights_;

    std::vector<LabelType> distinct_labels_;

};

template <typename RANDOMFOREST>
template <typename FEATURES, typename LABELS>
void GloballyRefinedRandomForest<RANDOMFOREST>::train(
        FEATURES const & features,
        LABELS const & labels
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "GloballyRefinedRandomForest::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "GloballyRefinedRandomForest::train(): Wrong label type.");

    vigra_precondition(rf_.num_classes() == 2,
                       "GloballyRefinedRandomForest::train(): Curently only implemented for binary random forests.");

    size_t const num_instances = features.shape()[0];

    // Create an adaptor for the graph structure and the node splits.
    std::vector<TreeGraph> tree_graphs;
    for (Tree const & tree : rf_.trees())
    {
        tree_graphs.push_back(tree.get_graph());
    }
    rf_adaptor_.set_forest(tree_graphs);

    // Get the leaf nodes of the training instances.
    MultiArray<2, size_t> leaf_ids(num_instances, rf_.num_trees());
    rf_.leaf_ids(features, leaf_ids);

    // Create the index vectors (= features) for the SVM.
    SparseFeatureGetter<UInt8> svm_features(Shape2(num_instances, rf_adaptor_.numLeaves()));
    for (size_t i = 0; i < num_instances; ++i)
    {
        for (size_t j = 0; j < rf_.num_trees(); ++j)
        {
            TreeNode tree_node(leaf_ids(i, j));
            Node node = rf_adaptor_.tree_to_forest(j, tree_node);
            size_t node_index = rf_adaptor_.getLeafIndex(node);
            svm_features(i, node_index) = 1;
        }
    }

    // Train the SVM.
    // TODO: Use early stopping criteria.
    SVM::Options opt;
    opt.normalize_ = false;
    opt.bias_value_ = 0.; // do not use bias feature
    SVM svm(opt);
    svm.train(svm_features, labels);
    distinct_labels_ = std::vector<LabelType>(svm.distinct_labels().begin(), svm.distinct_labels().end());

    // Save the produced leaf weights.
    svm_weights_.resize(rf_.num_trees());
    auto const & beta = svm.beta();
    for (size_t i = 0; i+1 < beta.size(); ++i) // do not use bias feature -> i+1 in termination condition
    {
        Node n = rf_adaptor_.getLeafNode(i);
        size_t ti;
        TreeNode tn;
        rf_adaptor_.forest_to_tree(n, ti, tn);
        svm_weights_[ti][tn] = beta(i);
    }
}

template <typename RANDOMFOREST>
template <typename FEATURES, typename LABELS>
void GloballyRefinedRandomForest<RANDOMFOREST>::predict(
        FEATURES const & features,
        LABELS & pred_y
) const {

    size_t const num_instances = features.shape()[0];

    // Get the leaf nodes of the training instances.
    MultiArray<2, size_t> leaf_ids(num_instances, rf_.num_trees());
    rf_.leaf_ids(features, leaf_ids);

    // Do the SVM prediction.
    for (size_t i = 0; i < num_instances; ++i)
    {
        double v = 0.;
        for (size_t j = 0; j < rf_.num_trees(); ++j)
        {
            TreeNode tree_node(leaf_ids(i, j));
            v = v + svm_weights_[j].at(tree_node);
        }
        size_t const index = (v >= 0) ? 0 : 1;
        pred_y(i) = distinct_labels_[index];
    }
}



}

#endif
