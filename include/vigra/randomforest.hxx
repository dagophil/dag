#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <queue>

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

template <typename T>
class FeatureGetter
{
public:
    typedef T value_type;

    FeatureGetter(MultiArrayView<2, T> const & arr)
        : arr_(arr)
    {}

    /// \brief Return the feature vector for instance i.
    /// \todo Are there problems with returning this MultiArrayView? Can you return a reference?
    MultiArrayView<1, T> operator[](size_t i)
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the const feature vector for instance i.
    /// \todo Are there problems with returning this MultiArrayView? Can you return a const reference?
    MultiArrayView<1, T> const operator[](size_t i) const
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the number of instances.
    size_t size() const
    {
        return arr_.shape()[0];
    }
protected:
    MultiArrayView<2, T> const & arr_;
};

template <typename T>
class LabelGetter
{
public:
    typedef T value_type;

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
    typedef Range::Iter Iter;

    Iter begin = instance_ranges_[node].begin;
    Iter end = instance_ranges_[node].end;
    size_t num_instances = std::distance(begin, end);

    // TODO: Remove output.
    std::cout << "splitting node " << node << " with "
              << num_instances << " instances" << std::endl;

    // Check whether the given node is pure.
    {
        bool is_pure = true;
        Iter it(begin);
        auto first_label = data_y[*it];
        for (Iter it(begin); it != end; ++it)
        {
            if (data_y[*it] != first_label)
            {
                is_pure = false;
                break;
            }
        }
        if (is_pure)
        {
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

    // TODO: Get a random subset of the features.
    // TODO: Find the best split.
    // TODO: Put instances into child nodes (according to best split). (Use std::partition and update the instance_ranges_ property map.)
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
        instance_indices_[i].reserve(data_x.size());
        for (size_t k = 0; k < data_x.size(); ++k)
            instance_indices_[i].push_back(k);
        // TODO: Can the loop be replaced with an equally efficient std::iota?

        Node rootnode = forest_.addNode();
        instance_ranges_[rootnode] = {instance_indices_[i].begin(), instance_indices_[i].end()};

        node_queue.push(rootnode);
    }

    while (!node_queue.empty())
    {
        Node node = node_queue.front();
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
