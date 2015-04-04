#include <iostream>
#include <vigra/dagraph.hxx>
#include <vector>
#include <utility>

void test_graph()
{
    std::vector<std::pair<int, int> > arcs {
        {0, 3},
        {0, 5},
        {3, 4},
        {5, 7}
    };

    vigra::StaticDAGraph0 g = vigra::StaticDAGraph0::build(8, arcs);
    g.print();
    g.print_root_nodes();
    g.print_leaf_nodes();

    std::cout << "Outgoing arcs of node 0:" << std::endl;
    vigra::StaticDAGraph0::Node n(0);
    for (vigra::StaticDAGraph0::OutArcIt it(g, n); it != lemon::INVALID; ++it)
    {
        std::cout << g.source(it).id() << " -> " << g.target(it).id() << std::endl;
    }

    std::cout << "Incoming arcs of node 3:" << std::endl;
    vigra::StaticDAGraph0::Node m(3);
    for (vigra::StaticDAGraph0::InArcIt it(g, m); it != lemon::INVALID; ++it)
    {
        std::cout << g.source(it).id() << " -> " << g.target(it).id() << std::endl;
    }
}

void test_tree()
{
    std::vector<std::pair<int, int> > arcs {
        {0, 3},
        {0, 5},
        {3, 4},
        {5, 7}
    };

    vigra::StaticForest0 g = vigra::StaticForest0::build(8, arcs);
    g.print();
    g.print_root_nodes();
    g.print_leaf_nodes();

    std::cout << "Outgoing arcs of node 0:" << std::endl;
    vigra::StaticForest0::Node n(0);
    for (vigra::StaticForest0::OutArcIt it(g, n); it != lemon::INVALID; ++it)
    {
        std::cout << g.source(it).id() << " : " << g.target(it).id() << std::endl;
    }

    std::cout << "Incoming arcs of node 3:" << std::endl;
    vigra::StaticForest0::Node m(3);
    for (vigra::StaticForest0::InArcIt it(g, m); it != lemon::INVALID; ++it)
    {
        std::cout << g.source(it).id() << " : " << g.target(it).id() << std::endl;
    }

    vigra::StaticForest0::Node p(7);
    g.parent(p);
    std::cout << "Parent of node 7: " << p.id() << std::endl;
}

int main()
{
    //test_graph();
    test_tree();
}
