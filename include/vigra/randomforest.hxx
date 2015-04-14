#ifndef VIGRA_RANDOMFOREST_HXX
#define VIGRA_RANDOMFOREST_HXX

#include "dagraph.hxx"

// TODO: Avoid naming conflicts in classes and file names.

namespace vigra
{

template <typename FOREST>
class RandomForestBase0
{
public:

    typedef FOREST Forest;

    RandomForestBase0();

    RandomForestBase0(RandomForestBase0 const &) = default;
    RandomForestBase0(RandomForestBase0 &&) = default;
    ~RandomForestBase0() = default;
    RandomForestBase0 & operator=(RandomForestBase0 const &) = default;
    RandomForestBase0 & operator=(RandomForestBase0 &&) = default;

    void train(size_t num_trees);


protected:

    size_t num_trees_;
    Forest forest_;
};

template <typename FOREST>
RandomForestBase0<FOREST>::RandomForestBase0()
    : num_trees_(0),
      forest_()
{
}

/*template <typename FOREST>
void RandomForestBase0<FOREST>::train(
        size_t num_trees,
        std::vector<size_t> const & instance_ids
){

}*/

}

#endif
