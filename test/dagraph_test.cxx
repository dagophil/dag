#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include <vigra/dagraph.hxx>
#include <vigra/hdf5impex.hxx>



void test_dagraph0()
{
    using namespace vigra;

    typedef DAGraph0 Graph;
    typedef Graph::Node Node;
    typedef Graph::Arc Arc;
    typedef Graph::NodeIt NodeIt;
    typedef Graph::ArcIt ArcIt;
    typedef Graph::OutArcIt OutArcIt;
    typedef Graph::InArcIt InArcIt;
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
            iter_nodes.push_back(Node(it));
        vigra_assert(nodes.size() == iter_nodes.size(), "Number of nodes incorrect.");
        std::sort(nodes.begin(), nodes.end());
        std::sort(iter_nodes.begin(), iter_nodes.end());
        for (size_t i = 0; i < nodes.size(); ++i)
            vigra_assert(nodes[i] == iter_nodes[i], "The node ids differ.");
    }

    // Check that the arc iterator walks over all arcs.
    // This tests not only the iterator but also the graph functions first and next.
    {
        std::vector<Arc> arcs {e0, e1, e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs incorrect.");
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Check that the out-arc iterator walks over all outgoing arcs of a node.
    // This tests not only the iterator but also the graph functions firstOut and nextOut.
    {
        std::vector<Arc> arcs {e1, e2};
        std::vector<Arc> iter_arcs;
        for (OutArcIt it(g, b); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs incorrect.");
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Check that the in-arc iterator walks over all incoming arcs of a node.
    // This tests not only the iterator but also the graph functions firstIn and nextIn.
    {
        std::vector<Arc> arcs {e2, e3};
        std::vector<Arc> iter_arcs;
        for (InArcIt it(g, d); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs incorrect.");
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Check that the child iterator walks over all children of a node.
    {
        std::vector<Node> nodes {c, d};
        std::vector<Node> iter_nodes;
        for (ChildIt it(g, b); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(it));
        vigra_assert(nodes.size() == iter_nodes.size(), "Number of nodes incorrect.");
        std::sort(nodes.begin(), nodes.end());
        std::sort(iter_nodes.begin(), iter_nodes.end());
        for (size_t i = 0; i < nodes.size(); ++i)
            vigra_assert(nodes[i] == iter_nodes[i], "The node ids differ.");
    }

    // Test the erase function for nodes.
    {
        g.erase(c);

        // Check that the node iterator only walks over the remaining nodes.
        std::vector<Node> nodes {a, b, d, e};
        std::vector<Node> iter_nodes;
        for (NodeIt it(g); it != lemon::INVALID; ++it)
            iter_nodes.push_back(Node(it));
        vigra_assert(nodes.size() == iter_nodes.size(), "Number of nodes is incorrect.");
        std::sort(nodes.begin(), nodes.end());
        std::sort(iter_nodes.begin(), iter_nodes.end());
        for (size_t i = 0; i < nodes.size(); ++i)
            vigra_assert(nodes[i] == iter_nodes[i], "The node ids differ.");

        // All arcs from or to c should have been removed.
        // Check that the arc iterator only walks over the remaining arcs.
        std::vector<Arc> arcs {e0, e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs is incorrect.");
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    // Test the erase function for arcs.
    {
        g.erase(e0);
        std::vector<Arc> arcs{e2, e3};
        std::vector<Arc> iter_arcs;
        for (ArcIt it(g); it != lemon::INVALID; ++it)
            iter_arcs.push_back(Arc(it));
        vigra_assert(arcs.size() == iter_arcs.size(), "Number of arcs is incorrect.");
        std::sort(arcs.begin(), arcs.end());
        std::sort(iter_arcs.begin(), iter_arcs.end());
        for (size_t i = 0; i < arcs.size(); ++i)
            vigra_assert(arcs[i] == iter_arcs[i], "The arc ids differ.");
    }

    std::cout << "test_dagraph0(): Success!" << std::endl;
}

void test_forest0()
{
    using namespace vigra;

    typedef Forest0 Forest;
    typedef Forest::Node Node;
    typedef Forest::Arc Arc;


    // Create the graph.
    Forest g;
    Node a = g.addNode();
    Node b = g.addNode();
    Node c = g.addNode();
    Node d = g.addNode();
    Arc e0 = g.addArc(a, b);
    Arc e1 = g.addArc(b, c);
    Arc e2 = g.addArc(b, d);

    // Test the parent() function.
    {
        Node tmp(d);
        g.parent(tmp);
        vigra_assert(tmp == b, "Error in Forest0::parent().");
        tmp = c;
        g.parent(tmp);
        vigra_assert(tmp == b, "Error in Forest0::parent().");
        g.parent(tmp);
        vigra_assert(tmp == a, "Error in Forest0::parent().");
        g.parent(tmp);
        vigra_assert(tmp == lemon::INVALID, "Error in Forest0::parent().");
    }

    // Test the is_root_node() function.
    {
        vigra_assert(g.is_root_node(a) && !g.is_root_node(b) && !g.is_root_node(c) && !g.is_root_node(d),
                     "Error in Forest0::is_root_node().");
    }

    std::cout << "test_forest0(): Success!" << std::endl;
}

void test_fixedforest0()
{
    using namespace vigra;

    typedef FixedForest0 Forest;
    typedef Forest::Node Node;
    typedef Forest::RootNodeIt RootNodeIt;
    typedef Forest::LeafNodeIt LeafNodeIt;

    // Create the graph.
    DAGraph0 g;
    std::vector<std::pair<int, int> > arcs {
        {0, 2},
        {0, 3},
        {1, 3},
        {2, 4},
        {2, 5}
    };
    for (size_t i = 0; i < 6; ++i)
        g.addNode();
    for (std::pair<int, int> const & a : arcs)
        g.addArc(Node(a.first), Node(a.second));

    // Create a forest from the graph.
    Forest f(g);

    // Check that the root node iterator walks over all root nodes.
    {
        std::vector<Node> roots {Node(0), Node(1)};
        std::vector<Node> iter_roots;
        for (RootNodeIt it(f); it != lemon::INVALID; ++it)
            iter_roots.push_back(Node(it));
        std::sort(iter_roots.begin(), iter_roots.end());
        vigra_assert(roots.size() == iter_roots.size(), "Number of root nodes is incorrect.");
        for (size_t i = 0; i < roots.size(); ++i)
            vigra_assert(roots[i] == iter_roots[i], "The root node ids differ.");
    }

    // Check that the leaf node iterator walks over all leaf nodes.
    {
        std::vector<Node> leaves {Node(3), Node(4), Node(5)};
        std::vector<Node> iter_leaves;
        for (LeafNodeIt it(f); it != lemon::INVALID; ++it)
            iter_leaves.push_back(Node(it));
        std::sort(iter_leaves.begin(), iter_leaves.end());
        vigra_assert(leaves.size() == iter_leaves.size(), "Number of leaf nodes is incorrect.");
        for (size_t i = 0; i < leaves.size(); ++i)
            vigra_assert(leaves[i] == iter_leaves[i], "The leaf node ids differ.");
    }

    std::cout << "test_fixedforest0(): Success!" << std::endl;
}

/*
template <typename S, typename T>
float exactness(
        vigra::MultiArrayView<2, S> const & train_x,
        vigra::MultiArrayView<1, T> const & train_y,
        vigra::MultiArrayView<2, S> const & test_x,
        vigra::MultiArrayView<1, T> const & test_y
){
    using namespace vigra;

    Forest0 forest;



    return 0;
}

void test_randomforest0()
{
    using namespace vigra;

    std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
    std::string test_filename = "/home/philip/data/ml-koethe/test.h5";

    // Read the training data.
    HDF5ImportInfo info(train_filename.c_str(), "images");
    MultiArray<3, float> train_images(Shape3(info.shape().begin()));
    readHDF5(info, train_images);
    MultiArrayView<2, float> train_images_tmp(Shape2(info.shape()[0]*info.shape()[1], info.shape()[2]), train_images.data());
    MultiArrayView<2, float> train_x(train_images_tmp.transpose());
//    std::cout << "Read train_x: " << train_x.shape() << std::endl;

    info = HDF5ImportInfo(train_filename.c_str(), "labels");
    MultiArray<1, UInt8> train_y(Shape1(info.shape().begin()));
    readHDF5(info, train_y);
//    std::cout << "Read train_y: " << train_y.shape() << std::endl;

    // Read the test data.
    info = HDF5ImportInfo(test_filename.c_str(), "images");
    MultiArray<3, float> test_images(Shape3(info.shape().begin()));
    readHDF5(info, test_images);
    MultiArrayView<2, float> test_images_tmp(Shape2(info.shape()[0]*info.shape()[1], info.shape()[2]), test_images.data());
    MultiArrayView<2, float> test_x(test_images_tmp.transpose());
//    std::cout << "Read test_x: " << test_x.shape() << std::endl;

    info = HDF5ImportInfo(test_filename.c_str(), "labels");
    MultiArray<1, UInt8> test_y(Shape1(info.shape().begin()));
    readHDF5(info, test_y);
//    std::cout << "Read test_y: " << test_y.shape() << std::endl;

    // Create small subsets for quicker tests.
    size_t small_size = 1000;
    MultiArrayView<2, float> train_x_small(train_x.subarray(Shape2(0, 0), Shape2(small_size, train_x.shape()[1])));
    MultiArrayView<1, UInt8> train_y_small(train_y.subarray(Shape1(0), Shape1(small_size)));
    MultiArrayView<2, float> test_x_small(test_x.subarray(Shape2(0, 0), Shape2(small_size, test_x.shape()[1])));
    MultiArrayView<1, UInt8> test_y_small(test_y.subarray(Shape1(0), Shape1(small_size)));

    float ex0 = exactness(train_x_small, train_y_small, test_x_small, test_y_small);
    std::cout << "Exactness: " << ex0 << std::endl;

    std::cout << "test_randomforest0(): Success!" << std::endl;
}
*/
int main()
{
    test_dagraph0();
    test_forest0();
    test_fixedforest0();
    //test_randomforest0();
}
