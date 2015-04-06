#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include <vigra/dagraph.hxx>



void test_dagraph0()
{
    using namespace vigra;

    typedef DAGraph0 Graph;
    typedef Graph::Node Node;
    typedef Graph::Arc Arc;
    typedef Graph::NodeIt NodeIt;
    typedef Graph::ArcIt ArcIt;

    // Create a graph.
    Graph g;
    Node a = g.addNode();
    Node b = g.addNode();
    Node c = g.addNode();
    Node d = g.addNode();
    Arc e0 = g.addArc(a, b);
    Arc e1 = g.addArc(b, c);
    Arc e2 = g.addArc(b, d);

    // Test the graph functions maxNodeId, maxArcId, source and target.
    {
        vigra_assert(g.maxNodeId() == 3, "Error in maxNodeId().");
        vigra_assert(g.maxArcId() == 2, "Error in maxArcId().");
        vigra_assert(g.source(e0) == a && g.source(e1) == b && g.source(e2) == b, "Error in source().");
        vigra_assert(g.target(e0) == b && g.target(e1) == c && g.target(e2) == d, "Error in target().");
    }

    // Check that the node iterator walks over all nodes.
    // This tests not only the iterator but also the graph functions first and next.
    {
        std::vector<Node> nodes {a, b, c, d};
        std::vector<Node> iter_nodes;
        for (NodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(it));
        std::sort(nodes.begin(), nodes.end());
        std::sort(iter_nodes.begin(), iter_nodes.end());
        vigra_assert(nodes.size() == iter_nodes.size(), "Number of nodes incorrect.");
        for (size_t i = 0; i < nodes.size(); ++i)
            vigra_assert(nodes[i] == iter_nodes[i], "The node ids differ.");
    }

    // Check that the arc iterator walks over all arcs.
    // This tests not only the iterator but also the graph functions first and next.
    {
        std::vector<Arc> arcs {e0, e1, e2};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs incorrect.");
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Test the erase function for nodes.
    {
        g.erase(c);
        std::vector<Node> nodes {a, b, d};
        std::vector<Node> iter_nodes;
        for (NodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(it));
        std::sort(nodes.begin(), nodes.end());
        std::sort(iter_nodes.begin(), iter_nodes.end());
        vigra_assert(nodes.size() == iter_nodes.size(), "Number of nodes is incorrect.");
        for (size_t i = 0; i < nodes.size(); ++i)
            vigra_assert(nodes[i] == iter_nodes[i], "The node ids differ.");

        std::vector<Arc> arcs {e0, e2};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs is incorrect.");
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Test the erase function for arcs.
    {
        g.erase(e0);
        std::vector<Arc> arcs{e2};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs is incorrect.");
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    std::cout << "test_dagraph0(): Success!" << std::endl;
}

void test_fixedforest0()
{
    using namespace vigra;

    typedef FixedForest0 Forest;
    typedef Forest::Node Node;
    typedef Forest::Arc Arc;
    typedef Forest::NodeIt NodeIt;
    typedef Forest::ArcIt ArcIt;

    Forest f;
    //Node a = f.addNode();

}

int main()
{
    test_dagraph0();
    test_fixedforest0();
}
