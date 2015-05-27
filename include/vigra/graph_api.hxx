#ifndef VIGRA_GRAPH_API_HXX
#define VIGRA_GRAPH_API_HXX

#include <vigra/graphs.hxx>

namespace vigra
{



namespace API
{

/**
 * \brief The IterGraph class.
 */
template <typename IDTYPE>
class IterDAG
{
public:

    typedef IDTYPE index_type;

    /**
     * \brief Node descriptor.
     *
     * The Node class is used as a node identifier (and not to store node data).
     * Graph functions may take Node objects as parameters and the graph itself
     * finds the appropriate data.
     */
    class Node
    {
    public:
        Node(lemon::Invalid = lemon::INVALID);
        bool operator!=(Node const & other) const;
        bool operator==(Node const & other) const;
        bool operator<(Node const & other) const;
    };

    /**
     * \brief Arc descriptor.
     *
     * The Arc class is used as an arc identifier (and not to store arc data).
     * Graph functions may take Arc objects as parameters and the graph itself
     * finds the appropriate data.
     */
    class Arc
    {
    public:
        Arc(lemon::Invalid = lemon::INVALID);
        bool operator!=(Arc const & other) const;
        bool operator==(Arc const & other) const;
        bool operator<(Arc const &) const;
    };

    /// \brief Iterator over all nodes.
    class NodeIt
    {
    public:
        NodeIt(lemon::Invalid = lemon::INVALID);
        NodeIt(IterDAG const & graph);
        NodeIt & operator++();
        Node operator*();
        bool operator!=(NodeIt const & other);
        bool operator==(NodeIt const & other);
    };

    /// \brief Iterator over all arcs.
    class ArcIt
    {
    public:
        ArcIt(lemon::Invalid = lemon::INVALID);
        ArcIt(IterDAG const & graph);
        ArcIt & operator++();
        Arc operator*();
        bool operator!=(ArcIt const & other);
        bool operator==(ArcIt const & other);
    };

    /// \brief Iterator over all outgoing arcs of a node.
    class OutArcIt
    {
    public:
        OutArcIt(IterDAG const & graph, Node const & node);
        OutArcIt & operator++();
        Arc operator*();
        bool operator!=(OutArcIt const & other);
        bool operator==(OutArcIt const & other);
    };

    /// \brief Iterator over all incoming arcs of a node.
    class InArcIt
    {
    public:
        InArcIt(IterDAG const & graph, Node const & node);
        InArcIt & operator++();
        Arc operator*();
        bool operator!=(InArcIt const & other);
        bool operator==(InArcIt const & other);
    };

    /**
     * \brief Iterator over all parents of a node.
     *
     * \note
     * This iterator exists only for convenience.
     * It could be emulated by taking the source of the InArcIt arcs.
     */
    class ParentIt
    {
    public:
        ParentIt(IterDAG const & graph, Node const & node);
        ParentIt & operator++();
        Node operator*();
        bool operator!=(ParentIt const & other);
        bool operator==(ParentIt const & other);
    };

    /**
     * \brief Iterator over all children of a node.
     *
     * \note
     * This iterator exists only for convenience.
     * It could be emulated by taking the target of the OutArcIt arcs.
     */
    class ChildIt
    {
    public:
        ChildIt(IterDAG const & graph, Node const & node);
        ChildIt & operator++();
        Node operator*();
        bool operator!=(ChildIt const & other);
        bool operator==(ChildIt const & other);
    };

    /// \brief Property map for nodes. Typically implemented with std::map.
    template <typename T>
    class NodeMap
    {
    public:
        typedef T value_type;
        typedef value_type & reference;
        typedef value_type const & const_reference;

        /// \brief Access the property of the given node.
        /// \note If the property for the given node does not exist, a runtime error is thrown.
        reference at(Node const & node);

        /// \brief Access the property of the given node.
        /// \note If the property for the given node does not exist, a runtime error is thrown.
        const_reference at(Node const & node) const;

        /// \brief Access the property of the given node.
        /// \note If the property for the given node does not exist, it is created with the default constructor.
        reference operator[](Node const & node);
    };

    /// \brief Property map for arcs. Typically implemented with std::map.
    template <typename T>
    class ArcMap
    {
    public:
        typedef T value_type;
        typedef value_type & reference;
        typedef value_type const & const_reference;

        /// \brief Access the property of the given arc.
        /// \note If the property for the given arc does not exist, an out_of_range exception is thrown.
        reference at(Arc const & arc);

        /// \brief Access the property of the given arc.
        /// \note If the property for the given arc does not exist, an out_of_range exception is thrown.
        const_reference at(Arc const & arc) const;

        /// \brief Access the property of the given arc.
        /// \note If the property for the given arc does not exist, it is created with the default constructor.
        reference operator[](Arc const & arc);
    };

    /// \brief Constructor for an empty graph.
    IterDAG();

    /// \brief Add a node.
    /// \todo This is not part of the lemon digraph. Why?
    Node addNode();

    /// \brief Add an arc from u to v.
    /// \todo This is not part of the lemon digraph. Why?
    Arc addArc(Node const & u, Node const & v);

    /// \brief Erase the given node.
    /// \todo This is not part of the lemon digraph. Why?
    void erase(Node const & node);

    /// \brief Erase the given arc.
    /// \todo This is not part of the lemon digraph. Why?
    void erase(Arc const & arc);

    /// \brief Return true if the node is present in the graph.
    bool valid(Node const & node) const;

    /// \brief Return true if the arc is present in the graph.
    bool valid(Arc const & arc) const;

    /// \brief Return the source node of the given arc.
    Node source(Arc const & arc) const;

    /// \brief Return the target node of the given arc.
    Node target(Arc const & arc) const;

    /// \brief Return the id of the given node.
    index_type id(Node const & node) const;

    /// \brief Return the id of the given arc.
    index_type id(Arc const & arc) const;

    /// \brief Return the node with the given id.
    Node nodeFromId(index_type const & id) const;

    /// \brief Return the arc with the given id.
    Arc arcFromId(index_type const & id) const;

    /// \brief Return the maximum node id of the currently saved nodes.
    index_type maxNodeId() const;

    /// \brief Return the number of nodes.
    size_t numNodes() const;

    /// \brief Return the number of arcs.
    size_t numArcs() const;

    /// \brief Return the in-degree of the node (= number of incoming arcs = number of parents).
    size_t inDegree(Node const & node) const;

    /// \brief Return the out-degree of the node (= number of outgoing arcs = number of children).
    size_t outDegree(Node const & node) const;

    /// \brief Return the number of parents of the given node.
    /// \note Convenience function for inDegree.
    size_t numParents(Node const & node) const;

    /// \brief Return the number of children of the given node.
    /// \note Convenience function for outDegree.
    size_t numChildren(Node const & node) const;

};

/**
 * \brief The RandomAccessDAG class.
 *
 * \note Even though it is not fully listed here, all of the IterDAG API is supported.
 *
 * \todo Once the IterDAG API is complete, list all of it here.
 */
class RandomAccessDAG
{
public:

    class Node {};
    class Arc {};

//    /// \brief Return the i-th node.
//    Node getNode(size_t i) const;

//    /// \brief Return the i-th arc.
//    Arc getArc(size_t i) const;

    /// \brief Return the i-th outgoing arc of the given node.
    Arc getOutArc(Node const & node, size_t i) const;

    /// \brief Return the i-th incoming arc of the given node.
    Arc getInArc(Node const & node, size_t i) const;

    /// \brief Return the i-th parent of the given node.
    Node getParent(Node const & node, size_t i = 0) const;

    /// \brief Return the i-th child of the given node.
    Node getChild(Node const & node, size_t i) const;

};

/**
 * \brief The SingleRootIterDAG class.
 *
 * One root node -> one connected component.
 *
 * \note Even though it is not fully listed here, all of the IterDAG API is supported.
 *
 * \todo Once the IterGraph API is complete, list all of it here.
 */
class SingleRootIterDAG
{
public:

    class Node {};

    /// \brief Iterator over all leaf nodes.
    class LeafIt
    {
    public:
        LeafIt(lemon::Invalid = lemon::INVALID);
        LeafIt(SingleRootIterDAG const & graph);
        LeafIt & operator++();
        Node operator*();
        bool operator!=(LeafIt const & other);
        bool operator==(LeafIt const & other);
    };

    /// \brief Return the root node.
    Node getRoot() const;

};

/**
 * \brief The SingleRootRandomAccessDAG class.
 *
 * \note Event though it is not full ylisted here, all of the RandomAccessDAG API is supported.
 *
 * \note Once the RandomAccessDAG API is complete, list all of it here.
 */
class SingleRootRandomAccessDAG
{
public:

    class Node {};

    /// \brief Return the number of leaf nodes.
    size_t numLeaves() const;

    /// \brief Return the i-th leaf node.
    Node getLeafNode(size_t i) const;

    /// \brief Return the root node.
    Node getRoot() const;

};

///**
// * \brief The IterJungle class.
// *
// * The IterJungle is a composite of single-root graphs. The IterDAG API is
// * fully supported, except that the addNode method needs an additional argument.
// */
//template <typename TREE>
//class IterJungle
//{
//public:

//    class Node {};

//    typedef TREE Tree;

//    typedef typename Tree::Node TreeNode;

//    typedef typename Tree::Arc TreeArc;

//    /// \brief Constructor for an empty jungle with the given number of trees.
//    IterJungle(size_t num_trees);

//    /// \brief Return the desired tree.
//    Tree & getTree(size_t i);

//    /// \brief Return the desired tree.
//    Tree const & getTree(size_t i) const;

//    /// \brief Add a node to the given tree.
//    Node addNode(size_t tree_index = 0);

//    /// \brief Return the tree index of the given node.
//    size_t getTreeIndex(Node const & node) const;

//};

///**
// * \brief The RandomAccessJungle class.
// *
// * The RandomAccessJungle is a composite of single-root graphs. The
// * RandomAccessDAG API is fully supported, except that the addNode method needs
// * an additional argument.
// */
//template <typename TREE>
//class RandomAccessJungle
//{
//public:

//    class Node {};

//    typedef TREE Tree;

//    typedef typename Tree::Node TreeNode;

//    typedef typename Tree::Arc TreeArc;

//    /// \brief Constructor for an empty jungle with the given number of trees.
//    RandomAccessJungle(size_t num_trees);

//    /// \brief Return the desired tree.
//    Tree & getTree(size_t i);

//    /// \brief Return the desired tree.
//    Tree const & getTree(size_t i) const;

//    /// \brief Add a node to the given tree.
//    Node addNode(size_t tree_index = 0);

//    /// \brief Return the tree index of the given node.
//    size_t getTreeIndex(Node const & node) const;
//};



} // namespace API



} // namespace vigra

#endif
