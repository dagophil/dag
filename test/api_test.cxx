#include <iostream>

#include <vigra/jungle.hxx>



void test_binary_tree()
{
    using namespace vigra;

    typedef BinaryTree Tree;
    typedef Tree::Node Node;
    typedef Tree::Arc Arc;

    // Create the tree.
    Tree tree;
    Node na = tree.addNode();
    Node nb = tree.addNode();
    Node nc = tree.addNode();
    Node nd = tree.addNode();
    Node ne = tree.addNode();
    Node nf = tree.addNode();
    Arc aa = tree.addArc(na, nb);
    Arc ab = tree.addArc(na, nc);
    Arc ac = tree.addArc(nc, nd);
    Arc ad = tree.addArc(nc, ne);
    Arc ae = tree.addArc(nd, nf);

    // Test the source and target methods.
    {
        vigra_assert(tree.source(aa) == na && tree.source(ab) == na && tree.source(ac) == nc &&
                     tree.source(ad) == nc && tree.source(ae) == nd,
                     "Error in BinaryTree::source().");
        vigra_assert(tree.target(aa) == nb && tree.target(ab) == nc && tree.target(ac) == nd &&
                     tree.target(ad) == ne && tree.target(ae) == nf,
                     "Error in BinaryTree.target().");
    }

    // Test numNodes and numArcs.
    {
        vigra_assert(tree.numNodes() == 6, "Error in BinaryTree.numNodes().");
        vigra_assert(tree.numArcs() == 5, "Error in BinaryTree.numArcs().");
    }

    // Test inDegree, outDegree, numParents and numChildren.
    // Since numParents calls inDegree and numChildren calls outDegree, we only test those methods.
    {
        vigra_assert(tree.inDegree(na) == 0 && tree.inDegree(nb) == 1 && tree.inDegree(nc) == 1 &&
                     tree.inDegree(nd) == 1 && tree.inDegree(ne) == 1 && tree.inDegree(nf) == 1,
                     "Error in BinaryTree::inDegree().");
        vigra_assert(tree.outDegree(na) == 2 && tree.outDegree(nb) == 0 && tree.outDegree(nc) == 2 &&
                     tree.outDegree(nd) == 1 && tree.outDegree(ne) == 0 && tree.outDegree(nf) == 0,
                     "Error in BinaryTree::outDegree().");
    }

    // Test getOutArc and getInArc.
    {
        vigra_assert(tree.getOutArc(na, 0) == aa && tree.getOutArc(na, 1) == ab && tree.getOutArc(nc, 0) == ac &&
                     tree.getOutArc(nc, 1) == ad && tree.getOutArc(nd, 0) == ae,
                     "Error in BinaryTree::getOutArc().");
        vigra_assert(tree.getInArc(nb, 0) == aa && tree.getInArc(nc, 0) == ab && tree.getInArc(nd, 0) == ac &&
                     tree.getInArc(ne, 0) == ad && tree.getInArc(nf, 0) == ae,
                     "Error in BinaryTree::getInArc().");
    }

    // Test getParent and getChild.
    {
        vigra_assert(tree.getParent(nb) == na && tree.getParent(nc) == na && tree.getParent(nd) == nc &&
                     tree.getParent(ne) == nc && tree.getParent(nf) == nd,
                     "Error in BinaryTree::getParent().");
        vigra_assert(tree.getChild(na, 0) == nb && tree.getChild(na, 1) == nc && tree.getChild(nc, 0) == nd &&
                     tree.getChild(nc, 1) == ne && tree.getChild(nd, 0) == nf,
                     "Error in BinaryTree::getChild().");
    }

    // Test numLeaves.
    {
        // TODO: Implement test.
    }

    // Test getLeafNode.
    {
        // TODO: Implement test.
    }

    // Test erase node.
    {
        tree.erase(nc);
        vigra_assert(tree.numNodes() == 5, "Error in BinaryTree::erase(Node).");
        vigra_assert(tree.numArcs() == 2, "Error in BinaryTree::erase(Node).");

        // TODO: Walk over all nodes and arcs and check that the correct ones are present.

        // Bring the graph back to the original state.
        nc = tree.addNode();
        ab = tree.addArc(na, nc);
        ac = tree.addArc(nc, nd);
        ad = tree.addArc(nc, ne);
        vigra_assert(tree.numNodes() == 6, "Error in BinaryTree::addNode().");
        vigra_assert(tree.numArcs() == 5, "Error in BinaryTree::addArc().");
    }

    // Test erase arc.
    {
        tree.erase(ab);
        vigra_assert(tree.numNodes() == 6, "Error in BinaryTree::erase(Arc).");
        vigra_assert(tree.numArcs() == 4, "Error in BinaryTree::erase(Arc).");

        // TODO: Walk over all nodes and arcs and check that the correct ones are present.

        // Bring the graph back to the original state.
        ab = tree.addArc(na, nc);
    }

    // Test getRoot.
    {
        vigra_assert(tree.getRoot() == na, "Error in BinaryTree::getRoot().");
        tree.erase(na);
        tree.erase(nb);
        vigra_assert(tree.getRoot() == nc, "Error in BinaryTree::getRoot().");

        // TODO: Add new nodes/arcs and test again.

        // Bring the graph back to the original state.
        na = tree.addNode();
        nb = tree.addNode();
        aa = tree.addArc(na, nb);
        ab = tree.addArc(na, nc);
    }

    // Test the property map.
    {
        Tree::NodeMap<int> map;
        map[na] = 6;
        map[nb] = 7;
        map[ne] = 1;
        vigra_assert(map[na] == 6 && map[nb] == 7 && map[ne] == 1, "Error in BinaryTree::NodeMap.");

        // TODO: Maybe improve this test.
    }

    std::cout << "test_binary_tree(): Success!" << std::endl;
}


int main()
{
    test_binary_tree();
}
