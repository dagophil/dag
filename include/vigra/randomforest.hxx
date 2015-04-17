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
    template <class ITER>
    struct IterRange
    {
    public:
        typedef ITER Iter;
        Iter begin;
        Iter end;
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

//    iterator begin()
//    {
//        return arr_.begin();
//    }

//    const_iterator begin() const
//    {
//        return arr_.begin();
//    }

//    iterator end()
//    {
//        return arr_.end();
//    }

//    const_iterator end() const
//    {
//        return arr_.end();
//    }
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

    /// \brief Split the given node and return the children. If no split was performed, n0 and n1 are invalid.
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

protected:

    /// \brief The forest structure.
    Forest forest_;

    /// \brief A property map with the labels of each pure node.
    PropertyMap<LabelType> labels_;

private:

    typedef detail::IterRange<std::vector<size_t>::iterator> Range;

    /// \brief Each tree has its instances saved in one of these vectors.
    std::vector<std::vector<size_t> > instance_indices_;

    /// \brief For each node, this map contains iterators to begin and end of the instances (instance_indices_).
    PropertyMap<Range> instance_ranges_;
};

template <typename FOREST, typename FEATURES, typename LABELS>
void RandomForest0<FOREST, FEATURES, LABELS>::split(
        Node const & node,
        FEATURES const & data_x,
        LABELS const & data_y,
        Node & n0,
        Node & n1
){
    auto const begin = instance_ranges_[node].begin;
    auto const end = instance_ranges_[node].end;
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
            std::cout << "added leaf with " << num_instances << " instances" << std::endl;
            labels_[node] = first_label;
            n0 = lemon::INVALID;
            n1 = lemon::INVALID;
            return;
        }
    }

    // Draw the bootstrap sample indices.
    std::vector<size_t> sample_indices;
    sample_indices.reserve(num_instances);
    detail::sample_with_replacement(num_instances, begin, end, std::back_inserter(sample_indices));

    // Get a random subset of the features.
    auto const num_feats = (size_t) std::ceil(std::sqrt(data_x.num_features()));
    std::vector<size_t> all_feat_indices;
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
    for (auto const feat : feat_indices)
    {
        // This map counts the labels of the instances that lie in the left child.
        std::map<LabelType, size_t> labels_left;

        // Sort the instances according to feature feat.
        auto const features_multiarray = data_x.get_features(feat);
        std::vector<std::pair<FeatureType, size_t> > features;
        features.reserve(sample_indices.size());
        for (size_t i = 0; i < sample_indices.size(); ++i)
            features.push_back({features_multiarray[sample_indices[i]], sample_indices[i]});
        std::sort(features.begin(), features.end());

        for (size_t i = 0; i+1 < features.size(); ++i)
        {
            // Add the new label to the left child.
            LabelType const new_label = data_y[features[i].second];
            auto it = labels_left.find(new_label);
            if (it == labels_left.end())
                labels_left[new_label] = 1;
            else
                ++(it->second);

            // Compute the gini impurity.
            float gini_left = 1;
            float gini_right = 1;
            float const n_left = i+1;
            float const n_right = features.size()-n_left;
            for (auto const & p : labels_left)
            {
                float p_left = p.second / n_left;
                float p_right = (label_priors[p.first] - p.second) / n_right;
                gini_left -= (p_left*p_left);
                gini_right -= (p_right*p_right);
            }
            float const gini = n_left * gini_left + n_right * gini_right;

            if (gini < best_gini)
            {
                best_gini = gini;
                best_split = (features[i].first + features[i+1].first) / 2;
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
    n0 = forest_.addNode();
    n1 = forest_.addNode();
    forest_.addArc(node, n0);
    forest_.addArc(node, n1);

    instance_ranges_[n0] = {begin, split_iter};
    instance_ranges_[n1] = {split_iter, end};
}

template <typename FOREST, typename FEATURES, typename LABELS>
void RandomForest0<FOREST, FEATURES, LABELS>::train(
        FEATURES const & data_x,
        LABELS const & data_y,
        size_t num_trees
){
    instance_indices_.resize(num_trees);

    std::queue<Node> node_queue;

    for (size_t i = 0; i < num_trees; ++i)
    {
        instance_indices_[i].reserve(data_x.num_instances());
        for (size_t k = 0; k < data_x.num_instances(); ++k)
            instance_indices_[i].push_back(k);
        // TODO: Can the loop be replaced with an equally efficient std::iota?

        auto const rootnode = forest_.addNode();
        instance_ranges_[rootnode] = {instance_indices_[i].begin(), instance_indices_[i].end()};

        node_queue.push(rootnode);
    }

    while (!node_queue.empty())
    {
        auto const node = node_queue.front();
        node_queue.pop();

        Node n0, n1;
        split(node, data_x, data_y, n0, n1);
        if (forest_.valid(n0))
            node_queue.push(n0);
        if (forest_.valid(n1))
            node_queue.push(n1);
    }
}



}

#endif
