#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
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
        typedef typename std::iterator_traits<ITER>::value_type ValueType;

        size_t num_instances = std::distance(begin, end);
        std::vector<ValueType> values(begin, end);
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
    template <typename LABELTYPE, typename COUNTTYPE>
    float gini_impurity(
            std::map<LABELTYPE, COUNTTYPE> const & labels_left,
            std::map<LABELTYPE, COUNTTYPE> const & label_priors
    ){
        typedef LABELTYPE LabelType;
        typedef COUNTTYPE CountType;
        typedef MapValueAdder<LabelType, CountType> Adder;

        CountType const n_total = std::accumulate(
                    label_priors.begin(),
                    label_priors.end(),
                    0,
                    Adder()
        );
        CountType const n_left = std::accumulate(
                    labels_left.begin(),
                    labels_left.end(),
                    0,
                    Adder()
        );
        CountType const n_right = n_total - n_left;

        float gini_left = 1;
        float gini_right = 1;
        for (auto const & p : labels_left)
        {
            auto const & label = p.first;
            auto const & count = p.second;
            float const p_left = count / static_cast<float>(n_left);
            float const p_right = (label_priors.at(label) - count) / static_cast<float>(n_right);
            gini_left -= (p_left*p_left);
            gini_right -= (p_right*p_right);
        }
        return n_left*gini_left + n_right*gini_right;
    }
}

/// \brief This class implements operator[] to return the feature vector of the requested instance.
template <typename T>
class FeatureGetter
{
public:
    typedef T value_type;

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

/// \brief Random forest class.
/// \note FEATURES/LABELS must implement the operator[] that gets an instance index and returns the features/labels of that instance.
template <typename FOREST, typename FEATURES, typename LABELS>
class RandomForest0
{
public:

    typedef FOREST Forest;
    typedef typename Forest::Node Node;
    typedef FEATURES Features;
    typedef typename Features::value_type FeatureType;
    typedef LABELS Labels;
    typedef typename Labels::value_type LabelType;

    template <typename VALUE_TYPE>
    using PropertyMap = typename Forest::template PropertyMap<VALUE_TYPE>;

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
            MultiArrayView<1, LabelType> & train_y
    ) const;

    /// \brief Split the given node and return the children. If no split was performed, n0 and n1 are invalid.
    /// \param tree_index: Index of the tree in the tree vector.
    /// \param node: The node that will be split.
    /// \param data_x: The features.
    /// \param data_y: The labels.
    /// \param[out] n0: The first child node.
    /// \param[out] n1: The second child node.
    void split(
            size_t tree_index,
            Node const & node,
            FEATURES const & data_x,
            LABELS const & data_y,
            Node & n0,
            Node & n1
    );

protected:

    // TODO: Use a struct that can hold a tree and its property maps.

    /// \brief The forest.
    std::vector<Forest> trees_;

    /// \brief Property maps with the node labels for each tree.
    std::vector<PropertyMap<LabelType> > node_labels_;

    /// \brief Property maps with the node splits for each tree.
    std::vector<PropertyMap<detail::Split<FeatureType> > > node_splits_;

private:

    typedef detail::IterRange<std::vector<size_t>::iterator> Range;

    /// \brief Each tree has its instances saved in one of these vectors.
    std::vector<std::vector<size_t> > instance_indices_;

    /// \brief Property maps with begin and end of the instances (instance_indices_) for each node in each tree.
    std::vector<PropertyMap<Range> > tree_instance_ranges_;
};

template <typename FOREST, typename FEATURES, typename LABELS>
void RandomForest0<FOREST, FEATURES, LABELS>::split(
        size_t tree_index,
        Node const & node,
        FEATURES const & data_x,
        LABELS const & data_y,
        Node & n0,
        Node & n1
){
    auto & instance_ranges = tree_instance_ranges_[tree_index];
    auto const begin = instance_ranges[node].begin;
    auto const end = instance_ranges[node].end;
    auto const num_instances = std::distance(begin, end);

//    // TODO: Remove output.
//    std::cout << "splitting node " << node << " with "
//              << num_instances << " instances" << std::endl;

    // Check whether the given node is pure.
    {
        auto is_pure = true;
        LabelType first_label = 0;
        if (num_instances > 1)
        {
            auto it(begin);
            first_label = data_y[*it];
            for (++it; it != end; ++it)
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
//            // TODO: Remove output.
//            std::cout << "added leaf with " << num_instances << " instances" << std::endl;
            node_labels_[tree_index][node] = first_label;
            n0 = lemon::INVALID;
            n1 = lemon::INVALID;
            return;
        }
    }

    // Draw the bootstrap sample indices.
    std::vector<size_t> sample_indices(begin, end);

    // Get a random subset of the features.
    auto const num_feats = (size_t) std::ceil(std::sqrt(data_x.num_features()));
    std::vector<size_t> all_feat_indices;
    all_feat_indices.reserve(data_x.num_features());
    for (size_t i = 0; i < data_x.num_features(); ++i)
        all_feat_indices.push_back(i);
    std::vector<size_t> feat_indices;
    feat_indices.reserve(num_feats);
    detail::sample_without_replacement(num_feats, all_feat_indices.begin(), all_feat_indices.end(), std::back_inserter(feat_indices));

    // Compute the prior label count.
    std::map<LabelType, size_t> label_priors;
    for (size_t i = 0; i < sample_indices.size(); ++i)
    {
        LabelType const l = data_y[sample_indices[i]];
        auto it = label_priors.find(l);
        if (it == label_priors.end())
            label_priors[l] = 1;
        else
            ++(it->second);
    }

    // Find the best split.
    size_t best_feat = 0;
    FeatureType best_split = 0;
    float best_gini = std::numeric_limits<float>::max();
    for (auto const & feat : feat_indices)
    {
        // Sort the instances according to the current feature.
        auto const features_multiarray = data_x.get_features(feat);
        std::sort(sample_indices.begin(), sample_indices.end(),
                [& features_multiarray](size_t a, size_t b){
                    return features_multiarray[a] < features_multiarray[b];
                }
        );

        // Compute the splits.
        std::vector<FeatureType> splits;
        for (size_t i = 0; i+1 < sample_indices.size(); ++i)
        {
            auto const & f0 = features_multiarray[sample_indices[i]];
            auto const & f1 = features_multiarray[sample_indices[i+1]];
            if (f0 != f1)
                splits.push_back((f0+f1)/2);
        }

        // This map keeps track of the labels of instances in the left child.
        std::map<LabelType, size_t> labels_left;

        size_t first_right_index = 0;
        for (auto const & s : splits)
        {
            // Add the new labels to the left child.
            do
            {
                LabelType const & new_label = data_y[sample_indices[first_right_index]];
                auto const it = labels_left.find(new_label);
                if (it == labels_left.end())
                    labels_left[new_label] = 1;
                else
                    ++(it->second);
                ++first_right_index;
            }
            while (features_multiarray[sample_indices[first_right_index]] < s);

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
    auto const split_iter = std::partition(begin, end,
            [&](size_t instance_index){
                return best_features[instance_index] < best_split;
            }
    );
    auto & tree = trees_[tree_index];
    n0 = tree.addNode();
    n1 = tree.addNode();
    tree.addArc(node, n0);
    tree.addArc(node, n1);
    instance_ranges[n0] = {begin, split_iter};
    instance_ranges[n1] = {split_iter, end};
    node_splits_[tree_index][node] = {best_feat, best_split};

//    // TODO: Remove output.
//    std::cout << "divided into " << std::distance(begin, split_iter) << " and " << std::distance(split_iter, end) << std::endl;
}

template <typename FOREST, typename FEATURES, typename LABELS>
void RandomForest0<FOREST, FEATURES, LABELS>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    trees_.resize(num_trees);
    instance_indices_.resize(num_trees);
    node_labels_.resize(num_trees);
    node_splits_.resize(num_trees);
    tree_instance_ranges_.resize(num_trees);

    for (size_t i = 0; i < num_trees; ++i)
    {
        std::queue<Node> node_queue;

        // Draw the bootstrap indices.
        std::vector<size_t> index_vector;
        index_vector.reserve(data_x.num_instances());
        for (size_t k = 0; k < data_x.num_instances(); ++k)
            index_vector.push_back(k);
        instance_indices_[i].reserve(data_x.num_instances());
        detail::sample_with_replacement(
                    data_x.num_instances(),
                    index_vector.begin(),
                    index_vector.end(),
                    std::back_inserter(instance_indices_[i])
        );

        // Add a new node to the graph and assign the bootstrap indices.
        auto & tree = trees_[i];
        auto const rootnode = tree.addNode();
        auto & instance_ranges = tree_instance_ranges_[i];
        instance_ranges[rootnode] = {instance_indices_[i].begin(), instance_indices_[i].end()};
        node_queue.push(rootnode);

        // Split the nodes.
        while (!node_queue.empty())
        {
            auto const node = node_queue.front();
            node_queue.pop();

            Node n0, n1;
            split(i, node, data_x, data_y, n0, n1);
            if (tree.valid(n0))
                node_queue.push(n0);
            if (tree.valid(n1))
                node_queue.push(n1);
        }
    }
}

template <typename FOREST, typename FEATURES, typename LABELS>
void RandomForest0<FOREST, FEATURES, LABELS>::predict(
        FEATURES const & test_x,
        MultiArrayView<1, LabelType> & train_y
) const {

    typedef typename Forest::RootNodeIt RootNodeIt;

    for (size_t i = 0; i < test_x.num_instances(); ++i)
    {
        auto const feats = test_x.instance_features(i);

        for (auto const & tree : trees_)
        {
            RootNodeIt it(tree);
            Node node(it);
            vigra_assert(tree.valid(node), "RandomForest0::predict(): The tree has no root node.");

            // TODO: Walk through the tree to a leaf node and get the node label.




        }
    }
}



}

#endif
