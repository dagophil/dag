#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include <vigra/multi_array.hxx>

#include "dagraph.hxx"

// TODO: Avoid naming conflicts in classes and file names.

namespace vigra
{

template <typename T>
class FeatureGetter
{
public:
    FeatureGetter(MultiArrayView<2, T> const & arr)
        : arr_(arr)
    {}

    /// \todo Are there problems with returning this MultiArrayView?
    MultiArrayView<1, T> operator[](size_t i)
    {
        return arr_.template bind<0>(i);
    }
protected:
    MultiArrayView<2, T> const & arr_;
};

template <typename FOREST>
class RandomForest0
{
public:

    typedef FOREST Forest;

    RandomForest0();

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

    size_t num_trees_;
    Forest forest_;
};

template <typename FOREST>
RandomForest0<FOREST>::RandomForest0()
    : num_trees_(0),
      forest_()
{
}

template <typename FOREST>
template <typename FEATURES, typename LABELS>
void RandomForest0<FOREST>::train(
        FEATURES const & train_x,
        LABELS const & train_y,
        size_t num_trees
){
    // TODO: Implement.
}



}

#endif
