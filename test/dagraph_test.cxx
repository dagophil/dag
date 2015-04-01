#include <iostream>
#include <vigra/dagraph.hxx>
#include <vector>
#include <utility>

int main()
{
    std::vector<std::pair<int, int> > arcs {
        {0, 3},
        {0, 5},
        {3, 4},
        {5, 7}
    };

    vigra::StaticDAGraph g = vigra::StaticDAGraph::build(8, arcs);
    g.print();

}
