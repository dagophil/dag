#ifndef VIGRA_GRAPH_API_HXX
#define VIGRA_GRAPH_API_HXX

#include <vigra/graphs.hxx>

namespace vigra
{



namespace API
{

namespace detail
{
    /**
     * \brief Node identifier.
     *
     * The Node class is used as a node identifier (and not to store node data).
     * Graph functions may take Node objects as parameters and the graph itself
     * finds the appropriate data.
     *
     * Two nodes are considered to be equal if the ids are equal.
     */
    template <typename ID>
    class Node
    {
    public:
        typedef ID index_type;
        Node(lemon::Invalid = lemon::INVALID);
        Node(index_type const & id);
        index_type id();
        void set_id(index_type const & id);
        bool operator!=(Node const & other);
        bool operator==(Node const & other);
    };

    /**
     * \brief Arc identifier.
     *
     * The Arc class is used as an arc identifier (and not to store arc data).
     * Graph functions may take Arc objects as parameters and the graph itself
     * finds the appropriate data.
     *
     * Two arcs are considered to be equal if the ids are equal.
     */
    template <typename ID>
    class Arc
    {
    public:
        typedef ID index_type;
        Arc(lemon::Invalid = lemon::INVALID);
        Arc(index_type const & id);
        index_type id();
        void set_id(index_type const & id);
        bool operator!=(Arc const & other);
        bool operator==(Arc const & other);
    };

    /// \brief Iterator over all nodes.
    template <typename GRAPH>
    class NodeIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        NodeIt(Graph const & graph);
        NodeIt & operator++();
        Node operator*();
        bool operator!=(NodeIt const & other);
        bool operator==(NodeIt const & other);
    };

    /// \brief Iterator over all arcs.
    template <typename GRAPH>
    class ArcIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Arc Arc;
        ArcIt(Graph const & graph);
        ArcIt & operator++();
        Arc operator*();
        bool operator!=(ArcIt const & other);
        bool operator==(ArcIt const & other);
    };

    /// \brief Iterator over all outgoing arcs of a node.
    template <typename GRAPH>
    class OutArcIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;
        OutArcIt(Graph const & graph, Node const & node);
        OutArcIt & operator++();
        Arc operator*();
        bool operator!=(OutArcIt const & other);
        bool operator==(OutArcIt const & other);
    };

    /// \brief Iterator over all incoming arcs of a node.
    template <typename GRAPH>
    class InArcIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;
        InArcIt(Graph const & graph, Node const & node);
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
    template <typename GRAPH>
    class ParentIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        ParentIt(Graph const & graph, Node const & node);
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
    template <typename GRAPH>
    class ChildIt
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        ChildIt(Graph const & graph, Node const & node);
        ChildIt & operator++();
        Node operator*();
        bool operator!=(ChildIt const & other);
        bool operator==(ChildIt const & other);
    };

    /// \brief Property map for nodes. Typically implemented with std::map.
    template <typename GRAPH, typename T>
    class NodeMap
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef T value_type;
        typedef value_type & reference;
        typedef value_type const & const_reference;

        /// \brief Access the property of the given node.
        /// \todo See operator[] above.
        const_reference operator[](Node const & node) const;

        /// \brief Access the property of the given node.
        /// \todo What is the preferred behavior when no property is stored for the given node? Create one with the default constructor or throw an exception?
        reference operator[](Node const & node);
    };

    /// \brief Property map for arcs. Typically implemented with std::map.
    template <typename GRAPH, typename T>
    class ArcMap
    {
    public:
        typedef GRAPH Graph;
        typedef typename Graph::Arc Arc;
        typedef T value_type;
        typedef value_type & reference;
        typedef value_type const & const_reference;

        /// \brief Access the property of the given arc.
        /// \todo What is the preferred behavior when no property is stored for the given arc? Create on with the default constructor or throw an exception?
        reference operator[](Arc const & arc);

        /// \brief Access the property of the given arc.
        /// \todo See operator[] above.
        const_reference operator[](Arc const & arc) const;
    };

} // namespace IterGraphDetail

/**
 * \brief The IterGraph class.
 */
template <typename ID = int>
class IterGraph
{
public:
    typedef ID index_type;
    typedef detail::Node<index_type> Node;
    typedef detail::Arc<index_type> Arc;
    typedef detail::NodeIt NodeIt;
    typedef detail::ArcIt ArcIt;
    typedef detail::OutArcIt OutArcIt;
    typedef detail::InArcIt InArcIt;
    typedef detail::ParentIt ParentIt;
    typedef detail::ChildIt ChildIt;

    /// \brief Default constructor.
    IterGraph();

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

    /// \brief Return the source node of the given arc.
    Node source(Arc const & arc) const;

    /// \brief Return the target node of the given arc.
    Node target(Arc const & arc) const;

    /// \brief Copy this graph into other.
    void copyTo(IterGraph & other) const;

private:

    /// \brief Not copy constructable. Use copyTo() instead.
    IterGraph(IterGraph const &);

    /// \brief Assignment is not allowed. Use copyTo() instead.
    void operator=(IterGraph const &);

};

/**
 * \brief The RandomAccessGraph class.
 *
 * \note Even though it is not fully listed here, all of the IterGraph API is supported.
 *
 * \todo Once the IterGraph API is complete, list all of it here.
 */
template <typename ID = int>
class RandomAccessGraph
{
public:

    typedef ID index_type;
    typedef detail::Node<index_type> Node;
    typedef detail::Arc<index_type> Arc;

    /// \brief Return the number of nodes.
    size_t numNodes() const;

    /// \brief Return the number of arcs.
    size_t numArcs() const;

    /// \brief Return the number of outgoing arcs of the given node.
    size_t numOutArcs(Node const & node);

    /// \brief Return the number of incoming arcs of the given node.
    size_t numInArcs(Node const & node);

    /// \brief Return the number of parents of the given node.
    size_t numParents(Node const & node);

    /// \brief Return the number of children of the given node.
    size_t numChildren(Node const & node);

    /// \brief Return the i-th node.
    Node getNode(size_t i);

    /// \brief Return the i-th arc.
    Arc getArc(size_t i);

    /// \brief Return the i-th outgoing arc of the given node.
    Arc getOutArc(Node const & node, size_t i);

    /// \brief Return the i-th incoming arc of the given node.
    Arc getInArc(node const & node, size_t i);

    /// \brief Return the i-th parent of the given node.
    Node getParent(Node const & node, size_t i);

    /// \brief Return the i-th child of the given node.
    Node getChild(Node const & node, size_t i);

};





} // namespace API



} // namespace vigra

#endif
