#ifndef VIGRA_GRAPH_API_HXX
#define VIGRA_GRAPH_API_HXX

namespace vigra
{



class IterGraphBase
{
public:

    /** \brief Node identifier.

     The Node class is used as a node identifier (and not to store node data).
     Graph functions may take Node objects as parameters and the graph itself
     finds the appropriate data.
     */
    class Node
    {
    public:
        bool operator!=(Node const & other);
        bool operator==(Node const & other);
    };

    /** \brief Arc identifier.

     The Arc class is used as an arc identifier (and not to store arc data).
     Graph functions may take Arc objects as parameters and the graph itself
     finds the appropriate data.
     */
    class Arc
    {
    public:
        bool operator!=(Node const & other);
        bool operator==(Node const & other);
    };

    /// \brief Node iterator.
    class NodeIter
    {
    public:
        NodeIter & operator++();
        Node operator*();
        bool operator!=(NodeIter const & other);
        bool operator==(NodeIter const & other);
    };

    /// \brief Arc iterator.
    class ArcIter
    {
    public:
        ArcIter & operator++();
        Arc operator*();
        bool operator!=(ArcIter const & other);
        bool operator==(ArcIter const & other);
    };

    /// \brief Add a node.
    Node add_node();

    /// \brief Add an arc from u to v.
    Arc add_arc(Node const & u, Node const & v);

    /// \brief Erase the given node.
    void erase(Node const & node);

    /// \brief Erase the given arc.
    void erase(Arc const & arc);

    /// \brief Return the source node of the given arc.
    Node source(Arc const & arc) const;

    /// \brief Return the target node of the given arc.
    Node target(Arc const & arc) const;

    /// \brief Return the begin node iterator.
    NodeIter nodes_begin() const;

    /// \brief Return the end node iterator.
    NodeIter nodes_end() const;

    /// \brief Return the begin arc iterator.
    ArcIter arcs_begin() const;

    /// \brief Return the end arc iterator.
    ArcIter arcs_end() const;

};



}

#endif
