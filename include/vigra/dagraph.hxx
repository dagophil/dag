#ifndef VIGRA_DAGRAPH_HXX
#define VIGRA_DAGRAPH_HXX

#include <vector>
#include <utility>

#include <vigra/graphs.hxx>

namespace vigra
{



namespace detail
{

    template <typename GRAPH, typename ITEM>
    class ItemIt
    {
    public:

        typedef GRAPH Graph;
        typedef ITEM Item;

        ItemIt(Graph const & graph);

        ItemIt(Graph const & graph, Item const & item);

        ItemIt & operator++();

        Item & operator*();

        Item* operator->();

        bool valid() const;

    protected:

        Graph const & graph_;
        Item current_;
    };

    template <typename GRAPH, typename ITEM>
    ItemIt<GRAPH, ITEM>::ItemIt(
            const Graph & graph
    )   : graph_(graph),
          current_()
    {
        graph_.first(current_);
    }

    template <typename GRAPH, typename ITEM>
    ItemIt<GRAPH, ITEM>::ItemIt(
            const Graph & graph,
            const Item & item
    )   : graph_(graph),
          current_(item)
    {}

    template <typename GRAPH, typename ITEM>
    ItemIt<GRAPH, ITEM> & ItemIt<GRAPH, ITEM>::operator++()
    {
        graph_.next(current_);
        return *this;
    }

    template <typename GRAPH, typename ITEM>
    typename ItemIt<GRAPH, ITEM>::Item & ItemIt<GRAPH, ITEM>::operator*()
    {
        return current_;
    }

    template <typename GRAPH, typename ITEM>
    typename ItemIt<GRAPH, ITEM>::Item* ItemIt<GRAPH, ITEM>::operator->()
    {
        return &current_;
    }

    template <typename GRAPH, typename ITEM>
    bool ItemIt<GRAPH, ITEM>::valid() const
    {
        return current_ != lemon::INVALID;
    }

    template <typename GRAPH, typename ITEM>
    bool operator==(ItemIt<GRAPH, ITEM> const & it, lemon::Invalid)
    {
        return !it.valid();
    }

    template <typename GRAPH, typename ITEM>
    bool operator==(lemon::Invalid, ItemIt<GRAPH, ITEM> const & it)
    {
        return !it.valid();
    }

    template <typename GRAPH, typename ITEM>
    bool operator!=(ItemIt<GRAPH, ITEM> const & it, lemon::Invalid)
    {
        return it.valid();
    }

    template <typename GRAPH, typename ITEM>
    bool operator!=(lemon::Invalid, ItemIt<GRAPH, ITEM> const & it)
    {
        return it.valid();
    }



    /// \brief Wrapper class for int, CLASS_ID can be used to create multiple classes that share the same code.
    template<typename INDEX_TYPE, int CLASS_ID>
    class GenericGraphItem
    {
    public:

        typedef INDEX_TYPE index_type;

        GenericGraphItem(const lemon::Invalid & iv = lemon::INVALID)
            : id_(-1)
        {}

        GenericGraphItem(const index_type id  )
            : id_(id)
        {}

        bool operator == (const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other ) const {
            return id_ == other.id_;
        }

        bool operator != (const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other ) const {
            return id_ != other.id_;
        }

        bool operator < (const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other ) const {
            return id_ < other.id_;
        }

        bool operator > (const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other ) const {
            return id_ > other.id_;
        }

        index_type id() const {
            return id_;
        }

        void set_id(index_type const & id) {
            id_ = id;
        }

    protected:

        index_type id_;
    };
}



class DAGraph0
{

public:

    typedef Int64 index_type;
    typedef detail::GenericGraphItem<index_type, 0> Node;
    typedef detail::GenericGraphItem<index_type, 1> Arc;
    typedef detail::ItemIt<DAGraph0, Node> NodeIt;
    typedef detail::ItemIt<DAGraph0, Arc> ArcIt;

protected:

    struct NodeT
    {
        int prev;
        int next;
        int first_in;
        int first_out;
    };

    struct ArcT
    {
        int source;
        int target;
        int prev_in;
        int next_in;
        int prev_out;
        int next_out;
    };

    std::vector<NodeT> nodes_;
    std::vector<ArcT> arcs_;
    int first_node_;
    int first_free_node_;
    int first_free_arc_;

public:

    DAGraph0();

    /// \todo Which of the following are really needed? Are there problems with the defaults? Is inline necessary?
    DAGraph0(DAGraph0 const &) = default;
    DAGraph0(DAGraph0 &&) = default;
    ~DAGraph0() = default;
    DAGraph0 & operator=(DAGraph0 const &) = default;
    DAGraph0 & operator=(DAGraph0 &&) = default;

    int maxNodeId() const;

    int maxArcId() const;

    Node source(Arc const & arc) const;

    Node target(Arc const & arc) const;

    void first(Node & node) const;

    void next(Node & node) const;

    void first(Arc & arc) const;

    void next(Arc & arc) const;

    void firstOut(Arc & arc, Node const & node) const;

    void nextOut(Arc & arc) const;

    void firstIn(Arc & arc, Node const & node) const;

    void nextIn(Arc & arc) const;

    static Node nodeFromId(int id);

    static Arc arcFromId(int id);

    bool valid(Node const & n) const;

    bool valid(Arc const & a) const;

    Node addNode();

    Arc addArc(Node const & u, Node const & v);

    void erase(Node const & node);

    void erase(Arc const & arc);

};

inline int DAGraph0::maxNodeId() const
{
    return static_cast<int>(nodes_.size())-1;
}

inline int DAGraph0::maxArcId() const
{
    return static_cast<int>(arcs_.size())-1;
}

inline DAGraph0::Node DAGraph0::source(
        const Arc & arc
) const {
    return Node(arcs_[arc.id()].source);
}

inline DAGraph0::Node DAGraph0::target(
        const Arc & arc
) const {
    return Node(arcs_[arc.id()].target);
}

inline void DAGraph0::first(
        Node & node
) const {
    node.set_id(first_node_);
}

inline void DAGraph0::next(
        Node & node
) const {
    node.set_id(nodes_[node.id()].next);
}

inline void DAGraph0::first(
        Arc & arc
) const {
    int n;
    for (n = first_node_;
         n != -1 && nodes_[n].first_out == -1;
         n = nodes_[n].next) {}
    if (n == -1)
        arc.set_id(-1);
    else
        arc.set_id(nodes_[n].first_out);
}

inline void DAGraph0::next(
        Arc & arc
) const {
    if (arcs_[arc.id()].next_out != -1)
    {
        arc.set_id(arcs_[arc.id()].next_out);
    }
    else
    {
        int n;
        for (n = nodes_[arcs_[arc.id()].source].next;
             n != -1 && nodes_[n].first_out == -1;
             n = nodes_[n].next) {}
        if (n == -1)
            arc.set_id(-1);
        else
            arc.set_id(nodes_[n].first_out);
    }
}

inline void DAGraph0::firstOut(
        Arc & arc,
        Node const & node
) const {
    arc.set_id(nodes_[node.id()].first_out);
}

inline void DAGraph0::nextOut(
        Arc & arc
) const {
    arc.set_id(arcs_[arc.id()].next_out);
}

inline void DAGraph0::firstIn(
        Arc & arc,
        Node const & node
) const {
    arc.set_id(nodes_[node.id()].first_in);
}

inline void DAGraph0::nextIn(
        Arc & arc
) const {
    arc.set_id(arcs_[arc.id()].next_in);
}

inline DAGraph0::DAGraph0()
    : nodes_(),
      arcs_(),
      first_node_(-1),
      first_free_node_(-1),
      first_free_arc_(-1)
{}

inline DAGraph0::Node DAGraph0::nodeFromId(
        int id
){
    return Node(id);
}

inline DAGraph0::Arc DAGraph0::arcFromId(
        int id
){
    return Arc(id);
}

inline bool DAGraph0::valid(
        Node const & n
) const {
    return n.id() >= 0 && n.id() < static_cast<int>(nodes_.size()) && nodes_[n.id()].prev != -2;
}

inline bool DAGraph0::valid(
        Arc const & a
) const {
    return a.id() >= 0 && a.id() < static_cast<int>(arcs_.size()) && arcs_[a.id()].prev_in != -2;
}

inline DAGraph0::Node DAGraph0::addNode()
{
    int n;

    if (first_free_node_ == -1)
    {
        n = nodes_.size();
        nodes_.push_back(NodeT());
    }
    else
    {
        n = first_free_node_;
        first_free_node_ = nodes_[n].next;
    }

    nodes_[n].next = first_node_;
    if (first_node_ != -1)
        nodes_[first_node_].prev = n;
    first_node_ = n;
    nodes_[n].prev = -1;
    nodes_[n].first_in = -1;
    nodes_[n].first_out = -1;

    return Node(n);
}

inline DAGraph0::Arc DAGraph0::addArc(
        Node const & u,
        Node const & v
){
    int a;

    if (first_free_arc_ == -1)
    {
        a = arcs_.size();
        arcs_.push_back(ArcT());
    }
    else
    {
        a = first_free_arc_;
        first_free_arc_ = arcs_[a].next_in;
    }

    arcs_[a].source = u.id();
    arcs_[a].target = v.id();

    arcs_[a].next_out = nodes_[u.id()].first_out;
    if (nodes_[u.id()].first_out != -1)
        arcs_[nodes_[u.id()].first_out].prev_out = a;

    arcs_[a].next_in = nodes_[v.id()].first_in;
    if (nodes_[v.id()].first_in != -1)
        arcs_[nodes_[v.id()].first_in].prev_in = a;

    arcs_[a].prev_in = arcs_[a].prev_out = -1;
    nodes_[u.id()].first_out = nodes_[v.id()].first_in = a;
    return Arc(a);
}

inline void DAGraph0::erase(
        Node const & node
){
    int n = node.id();

    if (nodes_[n].next != -1)
        nodes_[nodes_[n].next].prev = nodes_[n].prev;

    if (nodes_[n].prev != -1)
        nodes_[nodes_[n].prev].next = nodes_[n].next;
    else
        first_node_ = nodes_[n].next;

    nodes_[n].next = first_free_node_;
    first_free_node_ = n;
    nodes_[n].prev = -2;
}

inline void DAGraph0::erase(
        Arc const & arc
){
    int a = arc.id();

    if (arcs_[a].next_in != -1)
        arcs_[arcs_[a].next_in].prev_in = arcs_[a].prev_in;

    if (arcs_[a].prev_in != -1)
        arcs_[arcs_[a].prev_in].next_in = arcs_[a].next_in;
    else
        nodes_[arcs_[a].target].first_in = arcs_[a].next_in;

    if (arcs_[a].next_out != -1)
        arcs_[arcs_[a].next_out].prev_out = arcs_[a].prev_out;

    if (arcs_[a].prev_out != -1)
        arcs_[arcs_[a].prev_out].next_out = arcs_[a].next_out;
    else
        nodes_[arcs_[a].source].first_out = arcs_[a].next_out;

    arcs_[a].next_in = first_free_arc_;
    first_free_arc_ = a;
    arcs_[a].prev_in = -2;
}




/*
 * =====================================
 * ===               OLD             ===
 * =====================================
 * */
/// \brief Base class for a static directed acyclic graph (static: no nodes or edges can be added/removed).
class StaticDAGraph0
{

protected:

    /// \brief Node class.
    /// \todo Replace this with vigra::detail::GenericNode<int> (vigra/graph_item_impl.hxx).
    class Node
    {
    private:
        friend class StaticDAGraph0;
        friend class StaticForest0;

    protected:

        /// \brief Node id (= index in nodes_).
        int id_;

    public:

        Node()
        {}

        explicit Node(int pid)
            : id_(pid)
        {}

        Node(lemon::Invalid)
            : id_(-1)
        {}

        int id() const
        {
            return id_;
        }

        bool operator==(Node const & other) const
        {
            return id_ == other.id_;
        }

        bool operator!=(Node const & other) const
        {
            return id_ != other.id_;
        }
    };

    /// \brief Arc class.
    /// \todo Replace this with vigra::detail::GenericArc<int> (vigra/graph_item_impl.hxx).
    class Arc
    {
    private:
        friend class StaticDAGraph0;

    protected:

        /// \brief Arc id (= index in arcs_).
        int id_;

    public:

        Arc()
        {}

        explicit Arc(int pid)
            : id_(pid)
        {}

        Arc(lemon::Invalid)
            : id_(-1)
        {}

        int id() const
        {
            return id_;
        }

        bool operator==(Arc const & other) const
        {
            return id_ == other.id_;
        }

        bool operator!=(Arc const & other) const
        {
            return id_ != other.id_;
        }
    };

    struct NodeT {
        /// \brief Arc id of the first incoming arc.
        int first_in;

        /// \brief Arc id of the first outgoing arc.
        int first_out;

        /// \brief Node id of the previous root node.
        int prev_root;

        /// \brief Node id of the next root node.
        int next_root;

        /// \brief Node id of the previous leaf node.
        int prev_leaf;

        /// \brief Node id of the next leaf node.
        int next_leaf;

        // NOTE: For a non-static graph, the node ids prev and next may be added.
    };

    struct ArcT {
        /// \brief Node id of the target node.
        int target;

        /// \brief Node id of the source node.
        int source;

        /// \brief Arc id of the previous incoming arc.
        int prev_in;

        /// \brief Arc id of the next incoming arc.
        int next_in;

        /// \brief Arc id of the previous outgoing arc.
        int prev_out;

        /// \brief Arc id of the next outgoing arc.
        int next_out;
    };

    /// \brief Node container, the node id matches the index.
    std::vector<NodeT> nodes_;

    /// \brief Index of the first root node.
    int first_root_node_;

    /// \brief Index of the first leaf node.
    int first_leaf_node_;

    /// \brief Arc container.
    std::vector<ArcT> arcs_;

public:

    typedef Node Node;
    typedef Arc Arc;
    typedef NodeT NodeT;
    typedef ArcT ArcT;

    class OutArcIt;
    class InArcIt;

    /// \brief Construct the graph from a vector of pairs.
    /// \param num_nodes: Number of nodes.
    /// \param arcs: The pairs in this vector give the ids of the nodes that are connected by an arc.
    static StaticDAGraph0 build(
            size_t num_nodes,
            std::vector<std::pair<int, int> > const & arcs
    );

    /// \brief Return the number of nodes.
    size_t numNodes() const
    {
        return nodes_.size();
    }

    /// \brief Return the number of arcs.
    size_t numArcs() const
    {
        return arcs_.size();
    }

    /// \brief Find the first outgoing arc of a given node.
    /// \param[out] a: The first outgoing arc of n.
    /// \param n: The node.
    /// \todo: Change the order of arguments (output last).
    void firstOut(Arc & a, Node const & n) const
    {
        a.id_ = nodes_[n.id_].first_out;
    }

    /// \brief Replace an outgoing arc by the next outgoing arc.
    /// \param[in/out] a: The arc.
    void nextOut(Arc & a) const
    {
        a.id_ = arcs_[a.id_].next_out;
    }

    /// \brief Find the first incoming arc of a given node.
    /// \param[out] a: The first incoming arc of n.
    /// \param n: The node.
    /// \todo: Change the order of arguments (output last).
    void firstIn(Arc & a, Node const & n) const
    {
        a.id_ = nodes_[n.id_].first_in;
    }

    /// \brief Replace an incoming arc by the next incoming arc.
    /// \param[in/out] a: The arc.
    void nextIn(Arc & a) const
    {
        a.id_ = arcs_[a.id_].next_in;
    }

    /// \brief Return the source node of an arc.
    /// \param a: The arc.
    Node source(Arc const & a)
    {
        return Node(arcs_[a.id_].source);
    }

    /// \brief Return the target node of an arc.
    /// \param a: The arc.
    Node target(Arc const & a)
    {
        return Node(arcs_[a.id_].target);
    }

    /// \brief Print the number of nodes and the arcs.
    void print() const
    {
        std::cout << "Number of nodes: " << nodes_.size() << std::endl;
        std::cout << "Arcs:" << std::endl;
        for (ArcT const & a : arcs_)
        {
            std::cout << a.source << " -> " << a.target << std::endl;
        }
    }

    /// \brief Print all root nodes.
    void print_root_nodes() const
    {
        std::cout << "Root nodes:";

        int current = first_root_node_;
        while (current != -1)
        {
            std::cout << " " << current;
            current = nodes_[current].next_root;
        }
        std::cout << std::endl;
    }

    /// \brief Print all leaf nodes.
    void print_leaf_nodes() const
    {
        std::cout << "Leaf nodes: ";

        int current = first_leaf_node_;
        while (current != -1)
        {
            std::cout << " " << current;
            current = nodes_[current].next_leaf;
        }
        std::cout << std::endl;
    }

protected:

    /// \brief Since you cannot add or remove elements, the default constructor will always yield an empty graph, and that's why it is hidden.
    StaticDAGraph0()
        : first_root_node_(-1),
          first_leaf_node_(-1)
    {
    }

};

class StaticDAGraph0::OutArcIt : public Arc
{
private:
    const StaticDAGraph0* graph_;
public:
    OutArcIt()
    {}

    OutArcIt(lemon::Invalid)
        : Arc(lemon::Invalid()),
          graph_(nullptr)
    {}

    OutArcIt(StaticDAGraph0 const & graph, Node const & n)
        : graph_(&graph)
    {
        graph_->firstOut(*this, n);
    }

    OutArcIt(StaticDAGraph0 const & graph, Arc const & arc)
        : Arc(arc),
          graph_(&graph)
    {}

    OutArcIt & operator++()
    {
        graph_->nextOut(*this);
        return *this;
    }
};

class StaticDAGraph0::InArcIt : public Arc
{
private:
    const StaticDAGraph0* graph_;
public:
    InArcIt()
    {}

    InArcIt(lemon::Invalid)
        : Arc(lemon::Invalid()),
          graph_(nullptr)
    {}

    InArcIt(StaticDAGraph0 const & graph, Node const & n)
        : graph_(&graph)
    {
        graph_->firstIn(*this, n);
    }

    InArcIt(StaticDAGraph0 const & graph, Arc const & arc)
        : Arc(arc),
          graph_(&graph)
    {}

    InArcIt & operator++()
    {
        graph_->nextIn(*this);
        return *this;
    }
};

StaticDAGraph0 StaticDAGraph0::build(
        size_t num_nodes,
        std::vector<std::pair<int, int> > const & arcs
){
    StaticDAGraph0 g;
    if (num_nodes == 0)
        return g;

    // Create a graph with only root and leaf nodes.
    g.nodes_.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        g.nodes_.push_back({-1, -1, i-1, i+1, i-1, i+1});
    }
    g.nodes_.front().prev_root = -1;
    g.nodes_.front().prev_leaf = -1;
    g.nodes_.back().next_root = -1;
    g.nodes_.back().next_leaf = -1;
    g.first_root_node_ = 0;
    g.first_leaf_node_ = 0;

    // Add the arcs to the graph.
    for (auto const & a : arcs)
    {
        if (a.first < 0 || a.first >= num_nodes || a.second < 0 || a.second >= num_nodes)
            throw std::runtime_error("StaticDAGraph::build(): Node index out of range.");

        int u_id = a.first;
        int v_id = a.second;
        NodeT & u = g.nodes_[u_id];
        NodeT & v = g.nodes_[v_id];

        int arcid = g.arcs_.size();
        ArcT arc;
        arc.source = u_id;
        arc.target = v_id;
        arc.prev_out = -1;
        arc.prev_in = -1;

        if (u.first_out == -1)
        {
            arc.next_out = -1;
        }
        else
        {
            ArcT & outarc = g.arcs_[u.first_out];
            vigra_assert(outarc.prev_out == -1, "The first arc must not have a predecessor.");
            outarc.prev_out = arcid;
            arc.next_out = u.first_out;
        }
        u.first_out = arcid;

        if (v.first_in == -1)
        {
            arc.next_in = -1;
        }
        else
        {
            ArcT & inarc = g.arcs_[v.first_in];
            vigra_assert(inarc.prev_in == -1, "The first arc must not have a precedessor.");
            inarc.prev_in = arcid;
            arc.next_in = v.first_in;
        }
        v.first_in = arcid;

        // Modify leaf descriptors of u, its predecessor and its successor.
        if (u.next_leaf != -1)
            g.nodes_[u.next_leaf].prev_leaf = u.prev_leaf;
        if (u.prev_leaf != -1)
            g.nodes_[u.prev_leaf].next_leaf = u.next_leaf;
        else if (u_id == g.first_leaf_node_)
            g.first_leaf_node_ = u.next_leaf;
        u.prev_leaf = -1;
        u.next_leaf = -1;

        // Modify root descriptors of v, its predecessor and its successor.
        if (v.next_root != -1)
            g.nodes_[v.next_root].prev_root = v.prev_root;
        if (v.prev_root != -1)
            g.nodes_[v.prev_root].next_root = v.next_root;
        else if (v_id == g.first_root_node_)
            g.first_root_node_ = v.next_root;
        v.prev_root = -1;
        v.next_root = -1;

        g.arcs_.push_back(arc);
    }

    return g;
}

/// \todo Make the StaticForest0 a subclass of StaticDAGraph0.
class StaticForest0
{

protected:

    typedef StaticDAGraph0 Graph;

    StaticForest0(StaticDAGraph0 && graph)
        : graph_(graph)
    {}

public:

    typedef Graph::Node Node;
    typedef Graph::Arc Arc;
    typedef Graph::NodeT NodeT;
    typedef Graph::ArcT ArcT;

    class OutArcIt : public Graph::OutArcIt
    {
    public:
        OutArcIt()
            : Graph::OutArcIt()
        {}

        OutArcIt(lemon::Invalid)
            : Graph::OutArcIt(lemon::Invalid())
        {}

        OutArcIt(StaticForest0 const & forest, Node const & n)
            : Graph::OutArcIt(forest.graph_, n)
        {}

        OutArcIt(StaticForest0 const & forest, Arc const & arc)
            : Graph::OutArcIt(forest.graph_, arc)
        {}

        OutArcIt & operator++()
        {
            Graph::OutArcIt::operator++();
            return *this;
        }
    };

    class InArcIt : public Graph::InArcIt
    {
    public:
        InArcIt()
            : Graph::InArcIt()
        {}

        InArcIt(lemon::Invalid)
            : Graph::InArcIt(lemon::Invalid())
        {}

        InArcIt(StaticForest0 const & forest, Node const & n)
            : Graph::InArcIt(forest.graph_, n)
        {}

        InArcIt(StaticForest0 const & forest, Arc const & arc)
            : Graph::InArcIt(forest.graph_, arc)
        {}

        InArcIt & operator++()
        {
            Graph::InArcIt::operator++();
            return *this;
        }
    };

    /// \brief Construct the forest from a vector of pairs.
    /// \param num_nodes: Number of nodes.
    /// \param arcs: The pairs in this vector give the ids of the nodes that are connected by an arc.
    static StaticForest0 build(
            size_t num_nodes,
            std::vector<std::pair<int, int> > const & arcs
    ){
        StaticForest0 f(std::move(Graph::build(num_nodes, arcs)));
        return f;
    }

    /// \brief Return the source node of the given arc.
    /// \param a: The arc.
    Node source(Arc const & a)
    {
        return graph_.source(a);
    }

    /// \brief Return the target node of the given arc.
    /// \param a: The arc.
    Node target(Arc const & a)
    {
        return graph_.target(a);
    }

    /// \brief Replace a node by its parent.
    /// \param[in/out] n: The node that will be replace by its parent. If n is a root node, an invalid node will be returned.
    void parent(Node & n)
    {
        InArcIt it(*this, n);
        n.id_ = graph_.source(it).id();
    }

    void print()
    {
        graph_.print();
    }

    void print_root_nodes()
    {
        graph_.print_root_nodes();
    }

    void print_leaf_nodes()
    {
        graph_.print_leaf_nodes();
    }

private:

    Graph graph_;

};



}

#endif // VIGRA_DAGRAPH_HXX
