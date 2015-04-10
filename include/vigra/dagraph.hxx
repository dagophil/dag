#ifndef VIGRA_DAGRAPH_HXX
#define VIGRA_DAGRAPH_HXX

#include <vector>
#include <utility>
#include <vigra/graphs.hxx>

namespace vigra
{

namespace detail
{

    /// \brief Wrapper class for int. CLASS_ID can be used to create multiple classes that share the same code, but are seen as distinct classes by the compiler.
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

        bool operator==(const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other) const
        {
            return id_ == other.id_;
        }

        bool operator!=(const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other) const
        {
            return id_ != other.id_;
        }

        bool operator<(const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other) const
        {
            return id_ < other.id_;
        }

        bool operator>(const GenericGraphItem<INDEX_TYPE, CLASS_ID> & other) const
        {
            return id_ > other.id_;
        }

        index_type id() const
        {
            return id_;
        }

        void set_id(index_type const & id)
        {
            id_ = id;
        }

    protected:

        index_type id_;
    };

    /// \todo Find out whether the use of this function confuses the user. (E. g. if item is an arc, this operator gives the arc id and not source/target node.)
    template <typename INDEX_TYPE, int CLASS_ID>
    std::ostream & operator << (std::ostream & os, GenericGraphItem<INDEX_TYPE, CLASS_ID> const & item)
    {
        return os << item.id();
    }



    /// \brief Iterator for graph items (e. g. node, arc, leaf nodes, ...).
    /// \note Like in lemon, the iterator is a subclass of ITEM, so it can be used without dereferencing.
    template <typename GRAPH, typename ITEM, typename FUNCTOR>
    class ItemIt : public ITEM
    {
    public:

        typedef GRAPH Graph;
        typedef ITEM Item;
        typedef FUNCTOR Functor;

        ItemIt(Graph const & graph)
            : ITEM(),
              functor_(graph)
        {
            functor_.first(static_cast<Item &>(*this));
        }

        ItemIt(lemon::Invalid)
            : ITEM(lemon::INVALID),
              functor_(nullptr)
        {}

        ItemIt & operator++()
        {
            functor_.next(static_cast<Item &>(*this));
            return *this;
        }

        ItemIt* operator->()
        {
            return this;
        }

        ItemIt & operator*()
        {
            return *this;
        }

    protected:

        Functor functor_;
    };

    /// \brief Functor for ItemIt to iterate over the nodes of a graph.
    template <typename GRAPH>
    struct NodeItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;

        NodeItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        NodeItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node & node)
        {
            graph_->first(node);
        }

        void next(Node & node)
        {
            graph_->next(node);
        }

    protected:

        Graph const * graph_;
    };

    /// \brief Functor for ItemIt to iterate over the arcs of a graph.
    template <typename GRAPH>
    struct ArcItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Arc Arc;

        ArcItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        ArcItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Arc & arc)
        {
            graph_->first(arc);
        }

        void next(Arc & arc)
        {
            graph_->next(arc);
        }

    protected:

        Graph const * graph_;
    };

    /// \brief Functor for ItemIt to iterate over the root nodes of a graph.
    template <typename GRAPH>
    struct RootNodeItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;

        RootNodeItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        RootNodeItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node & node)
        {
            graph_->first(node);
            if (!graph_->valid(node))
                return;
            Arc arc;
            graph_->firstIn(arc, node);
            while (graph_->valid(arc))
            {
                graph_->next(node);
                if (!graph_->valid(node))
                    return;
                graph_->firstIn(arc, node);
            }
        }

        void next(Node & node)
        {
            graph_->next(node);
            if (!graph_->valid(node))
                return;
            Arc arc;
            graph_->firstIn(arc, node);
            while (graph_->valid(arc))
            {
                graph_->next(node);
                if (!graph_->valid(node))
                    return;
                graph_->firstIn(arc, node);
            }
        }

    protected:

        Graph const * graph_;
    };

    /// \brief Functor for ItemIt to iterate over the root nodes of a graph
    /// \note The Graph must implement the roots_cbegin() and roots_cend() methods to return a const_iterator to a vector with the root nodes.
    template <typename GRAPH>
    struct RootNodeVectorItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::const_node_iterator const_iterator;

        RootNodeVectorItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        RootNodeVectorItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node & node)
        {
            it_ = graph_->roots_cbegin();
            if (it_ != graph_->roots_cend())
                node.set_id(it_->id());
            else
                node = lemon::INVALID;
        }

        void next(Node & node)
        {
            ++it_;
            if (it_ != graph_->roots_cend())
                node.set_id(it_->id());
            else
                node = lemon::INVALID;
        }

    protected:

        Graph const * graph_;
        const_iterator it_;
    };

    /// \brief Functor for ItemIt to iterate over the leaf nodes of a graph.
    template <typename GRAPH>
    struct LeafNodeVectorItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::const_node_iterator const_iterator;

        LeafNodeVectorItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        LeafNodeVectorItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node & node)
        {
            it_ = graph_->leaves_cbegin();
            if (it_ != graph_->leaves_cend())
                node.set_id(it_->id());
            else
                node = lemon::INVALID;
        }

        void next(Node & node)
        {
            ++it_;
            if (it_ != graph_->leaves_cend())
                node.set_id(it_->id());
            else
                node = lemon::INVALID;
        }

    protected:

        Graph const * graph_;
        const_iterator it_;
    };

    template <typename GRAPH, typename ITEM, typename ITERITEM, typename FUNCTOR>
    class SubItemIt : public ITERITEM
    {
    public:

        typedef GRAPH Graph;
        typedef ITEM Item;
        typedef ITERITEM IterItem;
        typedef FUNCTOR Functor;

        SubItemIt(Graph const & graph, Item const & item)
            : ITERITEM(),
              functor_(graph)
        {
            functor_.first(item, static_cast<IterItem &>(*this));
        }

        SubItemIt(lemon::Invalid)
            : ITERITEM(lemon::INVALID),
              functor_(nullptr)
        {}

        SubItemIt & operator++()
        {
            functor_.next(static_cast<IterItem &>(*this));
            return *this;
        }

        SubItemIt* operator->()
        {
            return this;
        }

        SubItemIt & operator*()
        {
            return *this;
        }

    protected:

        Functor functor_;
    };

    /// \brief Functor for SubItemIt to iterate over all outgoing arcs of a node.
    template <typename GRAPH>
    struct OutArcItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;

        OutArcItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        OutArcItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node const & node, Arc & arc)
        {
            graph_->firstOut(arc, node);
        }

        void next(Arc & arc)
        {
            graph_->nextOut(arc);
        }

    protected:

        Graph const * graph_;
    };

    /// \brief Functor for SubItemIt to iterate over all incoming arcs of a node.
    template <typename GRAPH>
    struct InArcItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;

        InArcItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        InArcItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node const & node, Arc & arc)
        {
            graph_->firstIn(arc, node);
        }

        void next(Arc & arc)
        {
            graph_->nextIn(arc);
        }

    protected:

        Graph const * graph_;
    };

    template <typename GRAPH>
    struct ChildItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;

        ChildItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        ChildItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node const & sourcenode, Node & childnode)
        {
            graph_->firstOut(arc_, sourcenode);
            if (arc_ != lemon::INVALID)
                childnode = graph_->target(arc_);
            else
                childnode = lemon::INVALID;
        }

        void next(Node & childnode)
        {
            graph_->nextOut(arc_);
            if (arc_ != lemon::INVALID)
                childnode = graph_->target(arc_);
            else
                childnode = lemon::INVALID;
        }

    protected:

        Graph const * graph_;
        Arc arc_;
    };

} // namespace detail



/// \brief Class for arbitrary graphs. Api and implementation is taken from lemon::ListDigraph.
/// \todo Keep the api but change the implementation, so it is not just like lemon.
class DAGraph0
{

public:

    typedef Int64 index_type;
    typedef detail::GenericGraphItem<index_type, 0> Node;
    typedef detail::GenericGraphItem<index_type, 1> Arc;
    typedef detail::ItemIt<DAGraph0, Node, detail::NodeItFunctor<DAGraph0> > NodeIt;
    typedef detail::ItemIt<DAGraph0, Node, detail::RootNodeItFunctor<DAGraph0> > RootNodeIt;
    typedef detail::ItemIt<DAGraph0, Arc, detail::ArcItFunctor<DAGraph0> > ArcIt;
    typedef detail::SubItemIt<DAGraph0, Node, Arc, detail::OutArcItFunctor<DAGraph0> > OutArcIt;
    typedef detail::SubItemIt<DAGraph0, Node, Arc, detail::InArcItFunctor<DAGraph0> > InArcIt;
    typedef detail::SubItemIt<DAGraph0, Node, Node, detail::ChildItFunctor<DAGraph0> > ChildIt;

    DAGraph0();

    /// \todo Which of the following are really needed? Are there problems with the defaults?
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

    static int id(Node const & node);

    static int id(Arc const & arc);

    static Node nodeFromId(int id);

    static Arc arcFromId(int id);

    bool valid(Node const & n) const;

    bool valid(Arc const & a) const;

    Node addNode();

    Arc addArc(Node const & u, Node const & v);

    void erase(Node const & node);

    void erase(Arc const & arc);

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

};

inline DAGraph0::DAGraph0()
    : nodes_(),
      arcs_(),
      first_node_(-1),
      first_free_node_(-1),
      first_free_arc_(-1)
{}

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
        arc = lemon::INVALID;
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
            arc = lemon::INVALID;
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

inline int DAGraph0::id(
        const Node & node
){
    return node.id();
}

inline int DAGraph0::id(
        const Arc & arc
){
    return arc.id();
}

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

    // erase all outgoing arcs
    Arc arc;
    firstOut(arc, node);
    while (arc != lemon::INVALID)
    {
        erase(arc);
        firstOut(arc, node);
    }

    // erase all incoming arcs
    firstIn(arc, node);
    while (arc != lemon::INVALID)
    {
        erase(arc);
        firstIn(arc, node);
    }

    // erase the node itself
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



/// \brief The forest class extends a graph by some rootnode and parent functions.
/// \todo Decide whether to check that the forest constraint "each node has only one parent" is fulfilled.
/// \todo Maybe replace inheritance by containment.
/// \todo Maybe use template parameter instead of DAGraph0.
class Forest0 : public DAGraph0
{
public:

    Forest0() = default;

    /// \todo Which of the following are really needed? Are there problems with the defaults?
    Forest0(Forest0 const &) = default;
    Forest0(Forest0 &&) = default;
    ~Forest0() = default;
    Forest0 & operator=(Forest0 const &) = default;
    Forest0 & operator=(Forest0 &&) = default;

    bool is_root_node(Node const & node) const;

    void parent(Node & node) const;
};

bool Forest0::is_root_node(
        const Node & node
) const {
    Arc tmp;
    firstIn(tmp, node);
    return tmp == lemon::INVALID;
}

void Forest0::parent(
        Node & node
) const {
    Arc tmp;
    firstIn(tmp, node);
    if (tmp != lemon::INVALID)
    {
        node = source(tmp);
    }
    else
    {
        node = lemon::INVALID;
    }
}



/// \todo Inherit from Forest0 instead of DAGraph0.
class FixedForest0 : public DAGraph0
{
private:

    typedef DAGraph0 Parent;

public:

    typedef detail::ItemIt<FixedForest0, Node, detail::RootNodeVectorItFunctor<FixedForest0> > RootNodeIt;
    typedef detail::ItemIt<FixedForest0, Node, detail::LeafNodeVectorItFunctor<FixedForest0> > LeafNodeIt;
    typedef std::vector<Node>::const_iterator const_node_iterator;

    /// \brief Create the forest from a given graph.
    FixedForest0(DAGraph0 const & graph);

    const_node_iterator roots_cbegin() const;

    const_node_iterator roots_cend() const;

    const_node_iterator leaves_cbegin() const;

    const_node_iterator leaves_cend() const;

    // Hide all functions that modify the graph.
    /// \todo Is this a good practice? I guess the "is-a"-relation of OO-programming is violated.
    Node addNode() = delete;
    Arc addArc(Node const & u, Node const & v) = delete;
    void erase(Node const & node) = delete;
    void erase(Arc const & arc) = delete;

protected:

    /// \todo Change vector to map, so nodes can safely be added / removed when a forest is built.

    /// \brief This vector contains all root nodes and is always sorted.
    std::vector<Node> roots_;

    /// \brief This vector contains all leaf nodes and is always sorted.
    std::vector<Node> leaves_;
};

FixedForest0::FixedForest0(
        const DAGraph0 & graph
)   : Parent(graph)
{
    Arc arc;
    for (NodeIt it(*this); it != lemon::INVALID; ++it)
    {
        this->firstIn(arc, it);
        if (arc == lemon::INVALID)
            roots_.push_back(it);
        this->firstOut(arc, it);
        if (arc == lemon::INVALID)
            leaves_.push_back(it);
    }
    std::sort(roots_.begin(), roots_.end());
    std::sort(leaves_.begin(), leaves_.end());
}

FixedForest0::const_node_iterator FixedForest0::roots_cbegin() const
{
    return roots_.cbegin();
}

FixedForest0::const_node_iterator FixedForest0::roots_cend() const
{
    return roots_.cend();
}

FixedForest0::const_node_iterator FixedForest0::leaves_cbegin() const
{
    return leaves_.cbegin();
}

FixedForest0::const_node_iterator FixedForest0::leaves_cend() const
{
    return leaves_.cend();
}

} // namespace vigra

#endif // VIGRA_DAGRAPH_HXX
