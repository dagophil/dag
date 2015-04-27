#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include <vigra/dagraph.hxx>



/// \brief Return true if the sorted versions of v0 and v1 are equal.
template <typename CONTAINER>
bool equal_after_sort(CONTAINER const & v0, CONTAINER const & v1)
{
    CONTAINER v0_cpy(v0);
    CONTAINER v1_cpy(v1);
    if (v0_cpy.size() != v1_cpy.size())
        return false;
    std::sort(v0_cpy.begin(), v0_cpy.end());
    std::sort(v1_cpy.begin(), v1_cpy.end());
    for (size_t i = 0; i < v0_cpy.size(); ++i)
        if (v0_cpy[i] != v1_cpy[i]) return false;
    return true;
}

void test_dagraph0()
{
    using namespace vigra;

    typedef DAGraph0 Graph;
    typedef Graph::Node Node;
    typedef Graph::Arc Arc;
    typedef Graph::NodeIt NodeIt;
    typedef Graph::RootNodeIt RootNodeIt;
    typedef Graph::LeafNodeIt LeafNodeIt;
    typedef Graph::ArcIt ArcIt;
    typedef Graph::OutArcIt OutArcIt;
    typedef Graph::InArcIt InArcIt;
    typedef Graph::ParentIt ParentIt;
    typedef Graph::ChildIt ChildIt;

    // Create the graph.
    Graph g;
    Node a = g.addNode();
    Node b = g.addNode();
    Node c = g.addNode();
    Node d = g.addNode();
    Node e = g.addNode();
    Arc e0 = g.addArc(a, b);
    Arc e1 = g.addArc(b, c);
    Arc e2 = g.addArc(b, d);
    Arc e3 = g.addArc(e, d);

    // Test the graph functions maxNodeId, maxArcId, source and target.
    {
        vigra_assert(g.maxNodeId() == 4, "Error in DAGraph0::maxNodeId().");
        vigra_assert(g.maxArcId() == 3, "Error in DAGraph0::maxArcId().");
        vigra_assert(g.source(e0) == a && g.source(e1) == b && g.source(e2) == b && g.source(e3) == e,
                     "Error in DAGraph0::source().");
        vigra_assert(g.target(e0) == b && g.target(e1) == c && g.target(e2) == d && g.target(e3) == d,
                     "Error in DAGraph0::target().");
    }

    // Check that the node iterator walks over all nodes.
    // This tests not only the iterator but also the graph functions first and next.
    {
        std::vector<Node> nodes {a, b, c, d, e};
        std::vector<Node> iter_nodes;
        for (NodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in NodeIt.");
    }

    // Check that the root node iterator walks over all root nodes.
    {
        std::vector<Node> nodes {a, e};
        std::vector<Node> iter_nodes;
        for (RootNodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in RootNodeIt.");
    }

    // Check that the leaf node iterator walks over all leaf nodes.
    {
        std::vector<Node> nodes {c, d};
        std::vector<Node> iter_nodes;
        for (LeafNodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in LeafNodeIt.");
    }

    // Check that the arc iterator walks over all arcs.
    // This tests not only the iterator but also the graph functions first and next.
    {
        std::vector<Arc> arcs {e0, e1, e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in ArcIt.");
    }

    // Check that the out-arc iterator walks over all outgoing arcs of a node.
    // This tests not only the iterator but also the graph functions firstOut and nextOut.
    {
        std::vector<Arc> arcs {e1, e2};
        std::vector<Arc> iter_arcs;
        for (OutArcIt it(g, b); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in OutArcIt.");
    }

    // Check that the in-arc iterator walks over all incoming arcs of a node.
    // This tests not only the iterator but also the graph functions firstIn and nextIn.
    {
        std::vector<Arc> arcs {e2, e3};
        std::vector<Arc> iter_arcs;
        for (InArcIt it(g, d); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in InArcIt.");
    }

    // Check that the parent iterator walks over all parents of a node.
    {
        std::vector<Node> nodes {b, e};
        std::vector<Node> iter_nodes;
        for (ParentIt it(g, d); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in ParentIt.");
    }

    // Check that the child iterator walks over all children of a node.
    {
        std::vector<Node> nodes {c, d};
        std::vector<Node> iter_nodes;
        for (ChildIt it(g, b); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in ChildIt.");
    }

    // Test the isRootNode function.
    {
        vigra_assert(g.isRootNode(a) && !g.isRootNode(b) && !g.isRootNode(c) && !g.isRootNode(d) && g.isRootNode(e),
                     "Error in isRootNode().");
    }

    // Test the isLeafNode function.
    {
        vigra_assert(!g.isLeafNode(a) && !g.isLeafNode(b) && g.isLeafNode(c) && g.isLeafNode(d) && !g.isLeafNode(e),
                     "Error in isLeafNode().");
    }

    // Test the parent function.
    {
        Node tmp(a);
        g.parent(tmp);
        vigra_assert(!g.valid(tmp), "Error in parent().");
        tmp = b;
        g.parent(tmp);
        vigra_assert(tmp == a, "Error in parent().");
        tmp = c;
        g.parent(tmp);
        vigra_assert(tmp == b, "Error in parent().");
        tmp = d;
        g.parent(tmp);
        vigra_assert(tmp == b || tmp == e, "Error in parent().");
        tmp = e;
        g.parent(tmp);
        vigra_assert(!g.valid(tmp), "Error in parent().");
    }

    // Test the child function.
    {
        Node tmp(a);
        g.child(tmp);
        vigra_assert(tmp == b, "Error in child().");
        tmp = b;
        g.child(tmp);
        vigra_assert(tmp == c || tmp == d, "Error in child().");
        tmp = c;
        g.child(tmp);
        vigra_assert(!g.valid(tmp), "Error in child().");
        tmp = d;
        g.child(tmp);
        vigra_assert(!g.valid(tmp), "Error in child().");
        tmp = e;
        g.child(tmp);
        vigra_assert(tmp == d, "Error in child().");
    }

    // Test the erase function for nodes.
    {
        g.erase(c);

        // Check that the node iterator only walks over the remaining nodes.
        std::vector<Node> nodes {a, b, d, e};
        std::vector<Node> iter_nodes;
        for (NodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in erase(Node).");

        // All arcs from or to c should have been removed.
        // Check that the arc iterator only walks over the remaining arcs.
        std::vector<Arc> arcs {e0, e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in erase(Node).");

        // Re-add the node and the arc for the other tests.
        c = g.addNode();
        e1 = g.addArc(b, c);
    }

    // Test the erase function for arcs.
    {
        g.erase(e0);
        std::vector<Arc> arcs{e1, e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in erase(Arc).");

        // Re-add the arc for the other tests.
        e0 = g.addArc(a, b);
   }

    std::cout << "test_dagraph0(): Success!" << std::endl;
}

void test_forest1()
{
    using namespace vigra;

    typedef DAGraph0 Graph;
    typedef Forest1<Graph> Forest;
    typedef Forest::Node Node;
    typedef Forest::Arc Arc;
    typedef Forest::NodeIt NodeIt;
    typedef Forest::ArcIt ArcIt;
    typedef Forest::RootNodeIt RootNodeIt;
    typedef Forest::LeafNodeIt LeafNodeIt;

    Forest g;
    Node a = g.addNode();
    Node b = g.addNode();
    Node c = g.addNode();
    Node d = g.addNode();
    Node e = g.addNode();
    Arc e0 = g.addArc(a, b);
    Arc e1 = g.addArc(b, c);
    Arc e2 = g.addArc(b, d);
    Arc e3 = g.addArc(d, e);

    // Test the std-like iterators.
    {
        std::vector<Node> nodes {a};
        std::vector<Node> iter_nodes;
        for (auto it = g.roots_cbegin(); it != g.roots_cend(); ++it)
            iter_nodes.push_back(*it);
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in roots_cbegin() or roots_cend().");
    }
    {
        std::vector<Node> nodes {c, e};
        std::vector<Node> iter_nodes;
        for (auto it = g.leaves_cbegin(); it != g.leaves_cend(); ++it)
            iter_nodes.push_back(*it);
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in leaves_cbegin() or leaves_cend().");
    }

    // Test the lemon-like iterators
    {
        std::vector<Node> nodes {a};
        std::vector<Node> iter_nodes;
        for (RootNodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in RootNodeIt.");
    }
    {
        std::vector<Node> nodes {c, e};
        std::vector<Node> iter_nodes;
        for (LeafNodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in LeafNodeIt.");
    }

    // Test the erase function for nodes.
    {
        g.erase(d);

        // Check that the root nodes are correct.
        std::vector<Node> roots {a, e};
        std::vector<Node> iter_roots;
        for (RootNodeIt it(g); it != lemon::INVALID; ++it)
            iter_roots.push_back(Node(*it));
        vigra_assert(equal_after_sort(roots, iter_roots), "Error in erase(Node).");

        // Check that the leaf nodes are correct.
        std::vector<Node> leaves {c, e};
        std::vector<Node> iter_leaves;
        for (LeafNodeIt it(g); it != lemon::INVALID; ++it)
            iter_leaves.push_back(Node(*it));
        vigra_assert(equal_after_sort(leaves, iter_leaves), "Error in erase(Node).");

        // Re-add the node and the arcs for the other tests.
        d = g.addNode();
        e2 = g.addArc(b, d);
        e3 = g.addArc(d, e);
    }

    // Test the erase function for arcs.
    {
        g.erase(e2);

        // Check that the root nodes are correct.
        std::vector<Node> roots {a, d};
        std::vector<Node> iter_roots;
        for (RootNodeIt it(g); it != lemon::INVALID; ++it)
            iter_roots.push_back(Node(*it));
        vigra_assert(equal_after_sort(roots, iter_roots), "Error in erase(Arc).");

        // Check that the leaf nodes are correct.
        std::vector<Node> leaves {c, e};
        std::vector<Node> iter_leaves;
        for (LeafNodeIt it(g); it != lemon::INVALID; ++it)
            iter_leaves.push_back(Node(*it));
        vigra_assert(equal_after_sort(leaves, iter_leaves), "Error in erase(Arc).");

        e2 = g.addArc(b, d);
    }

    // Test the construction from graph.
    {
        Graph gr;
        Node a = gr.addNode();
        Node b = gr.addNode();
        Node c = gr.addNode();
        Node d = gr.addNode();
        Node e = gr.addNode();
        Node f = gr.addNode();
        Node g = gr.addNode();
        Node h = gr.addNode();
        Arc e0 = gr.addArc(a, b);
        Arc e1 = gr.addArc(b, c);
        Arc e2 = gr.addArc(b, d);
        Arc e3 = gr.addArc(d, e);
        Arc e4 = gr.addArc(f, g);
        Arc e5 = gr.addArc(f, h);

        Forest fo(gr);

        std::vector<Node> nodes {a, b, c, d, e, f, g, h};
        std::vector<Node> iter_nodes;
        for (NodeIt it(fo); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(*it));
        vigra_assert(equal_after_sort(nodes, iter_nodes), "Error in constructor from graph.");

        std::vector<Arc> arcs {e0, e1, e2, e3, e4, e5};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(fo); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(*it));
        vigra_assert(equal_after_sort(arcs, iter_arcs), "Error in constructor from graph.");

        std::vector<Node> roots {a, f};
        std::vector<Node> iter_roots;
        for (RootNodeIt it(fo); it != lemon::INVALID; ++it)
            iter_roots.push_back(Node(*it));
        vigra_assert(equal_after_sort(roots, iter_roots), "Error in constructor from graph.");

        std::vector<Node> leaves {c, e, g, h};
        std::vector<Node> iter_leaves;
        for (LeafNodeIt it(fo); it != lemon::INVALID; ++it)
            iter_leaves.push_back(Node(*it));
        vigra_assert(equal_after_sort(leaves, iter_leaves), "Error in constructor from graph.");
    }

    std::cout << "test_forest1(): Success!" << std::endl;
}

int main()
{
    test_dagraph0();
    test_forest1();
}
