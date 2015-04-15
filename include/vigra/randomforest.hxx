#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>
#include <queue>

#include "dagraph.hxx"

// TODO: Avoid naming conflicts in classes and file names.

namespace vigra
{

namespace detail
{
    template <class ITER>
    struct IterRange
    {
    public:
        ITER instances_begin;
        ITER instances_end;
    };
}

template <typename T>
class FeatureGetter
{
public:
    FeatureGetter(MultiArrayView<2, T> const & arr)
        : arr_(arr)
    {}

    /// \brief Return the feature vector for instance i.
    /// \todo Are there problems with returning this MultiArrayView?
    MultiArrayView<1, T> operator[](size_t i)
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

template <typename FOREST>
class RandomForest0
{
public:

    typedef FOREST Forest;

    template <typename VALUE_TYPE>
    using PropertyMap = typename Forest::template PropertyMap<VALUE_TYPE>;

    RandomForest0() = default;
    RandomForest0(RandomForest0 const &) = default;
    RandomForest0(RandomForest0 &&) = default;
    ~RandomForest0() = default;
    RandomForest0 & operator=(RandomForest0 const &) = default;
    RandomForest0 & operator=(RandomForest0 &&) = default;

    /// \brief Train the random forest.
    /// \note FEATURES/LABELS must implement the operator[] that gets an instance index and returns the features/labels of that instance.
    template <typename FEATURES, typename LABELS>
    void train(
            FEATURES const & train_x,
            LABELS const & train_y,
            size_t num_trees
    );

protected:

    Forest forest_;
};

template <typename FOREST>
template <typename FEATURES, typename LABELS>
void RandomForest0<FOREST>::train(
        FEATURES const & train_x,
        LABELS const & train_y,
        size_t num_trees
){
    typedef typename Forest::Node Node;
    typedef detail::IterRange<std::vector<size_t>::iterator> Range;

    // Since instance_ranges contains iterators to the vectors in instance_indices,
    // instance_ranges and instance_indices must have the same lifetime!
    PropertyMap<Range> instance_ranges;
    std::vector<std::vector<size_t> > instance_indices(num_trees);

    std::queue<Node> node_queue;

    for (size_t i = 0; i < num_trees; ++i)
    {
        instance_indices[i].reserve(train_x.size());
        for (size_t k = 0; k < train_x.size(); ++k)
            instance_indices[i].push_back(k);
        // TODO: Maybe replace the loop with std::iota.

        Node rootnode = forest_.addNode();
        instance_ranges[rootnode] = {instance_indices[i].begin(), instance_indices[i].end()};

        node_queue.push(rootnode);
    }

    while (!node_queue.empty())
    {
        Node node = node_queue.front();
        node_queue.pop();

        // TODO: Split the node and add the children to the queue.

    }
}



}

#endif
