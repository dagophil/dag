#ifndef VIGRA_DAGRAPH_HXX
#define VIGRA_DAGRAPH_HXX

#include <vector>
#include <utility>
#include <unordered_set>
#include <map>

#include <vigra/graphs.hxx>  // for lemon::INVALID

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

        GenericGraphItem(const index_type id)
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

    template <typename INDEX_TYPE, int CLASS_ID>
    std::ostream & operator << (std::ostream & os, GenericGraphItem<INDEX_TYPE, CLASS_ID> const & item)
    {
        return os << item.id();
    }



    /// \brief Iterator for graph items (e. g. node, arc, leaf nodes, ...).
    /// \note Like in lemon, the iterator is a subclass of ITEM, so it can be used without dereferencing.
    template <typename GRAPH, typename FUNCTOR>
    class ItemIt : public FUNCTOR::Item
    {
    public:

        typedef GRAPH Graph;
        typedef FUNCTOR Functor;
        typedef typename Functor::Item Item;

        ItemIt(Graph const & graph)
            : Item(),
              functor_(graph)
        {
            functor_.first(static_cast<Item &>(*this));
        }

        ItemIt(lemon::Invalid)
            : Item(lemon::INVALID),
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
        typedef Node Item;  // needed to be used by ItemIt

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
        typedef Arc Item;  // needed to be used by ItemIt

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
        typedef Node Item;  // needed to be used by ItemIt

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

    /// \brief Functor for ItemIt to iterate over the leaf nodes of a graph.
    template <typename GRAPH>
    struct LeafNodeItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;
        typedef Node Item;  // needed to be used by ItemIt

        LeafNodeItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        LeafNodeItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node & node)
        {
            graph_->first(node);
            if (!graph_->valid(node))
                return;
            Arc arc;
            graph_->firstOut(arc, node);
            while (graph_->valid(arc))
            {
                graph_->next(node);
                if (!graph_->valid(node))
                    return;
                graph_->firstOut(arc, node);
            }
        }

        void next(Node & node)
        {
            graph_->next(node);
            if (!graph_->valid(node))
                return;
            Arc arc;
            graph_->firstOut(arc, node);
            while (graph_->valid(arc))
            {
                graph_->next(node);
                if (!graph_->valid(node))
                    return;
                graph_->firstOut(arc, node);
            }
        }

    protected:

        Graph const * graph_;
    };

    /// \brief Functor for ItemIt to iterate over the root nodes of a graph
    /// \note The Graph must implement the roots_cbegin() and roots_cend() methods to return a const_iterator to a vector with the root nodes.
    template <typename GRAPH, typename ITERATOR = typename GRAPH::const_iterator>
    struct RootNodeVectorItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef ITERATOR Iterator;
        typedef typename Graph::Node Node;
        typedef Node Item;  // needed to be used by ItemIt

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
        Iterator it_;
    };

    /// \brief Functor for ItemIt to iterate over the leaf nodes of a graph.
    /// \note The Graph must implement the leaves_cbegin() and leaves_cend() methods to return a const_iterator to a vector with the leaf nodes.
    template <typename GRAPH, typename ITERATOR = typename GRAPH::const_iterator>
    struct LeafNodeVectorItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef ITERATOR Iterator;
        typedef typename Graph::Node Node;
        typedef Node Item;  // needed to be used by ItemIt

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
        Iterator it_;
    };

    template <typename GRAPH, typename FUNCTOR>
    class SubItemIt : public FUNCTOR::IterItem
    {
    public:

        typedef GRAPH Graph;
        typedef FUNCTOR Functor;
        typedef typename Functor::Item Item;
        typedef typename Functor::IterItem IterItem;

        SubItemIt(Graph const & graph, Item const & item)
            : IterItem(),
              functor_(graph)
        {
            functor_.first(item, static_cast<IterItem &>(*this));
        }

        SubItemIt(lemon::Invalid)
            : IterItem(lemon::INVALID),
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
        typedef Node Item;  // needed to be used by SubItemIt
        typedef Arc IterItem;  // needed to be used by SubItemIt

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
        typedef Node Item;  // needed to be used by SubItemIt
        typedef Arc IterItem;  // needed to be used by SubItemIt

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

    /// \brief Functor for SubItemIt to iterate over all children of a node.
    template <typename GRAPH>
    struct ChildItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;
        typedef Node Item;  // needed to be used by SubItemIt
        typedef Node IterItem;  // needed to be used by SubItemIt

        ChildItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        ChildItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node const & sourcenode, Node & childnode)
        {
            graph_->firstOut(arc_, sourcenode);
            if (graph_->valid(arc_))
                childnode = graph_->target(arc_);
            else
                childnode = lemon::INVALID;
        }

        void next(Node & childnode)
        {
            graph_->nextOut(arc_);
            if (graph_->valid(arc_))
                childnode = graph_->target(arc_);
            else
                childnode = lemon::INVALID;
        }

    protected:

        Graph const * graph_;
        Arc arc_;
    };

    /// \brief Functor for SubItemIt to iterate over all parents of a node.
    template <typename GRAPH>
    struct ParentItFunctor
    {
    public:

        typedef GRAPH Graph;
        typedef typename Graph::Node Node;
        typedef typename Graph::Arc Arc;
        typedef Node Item;  // needed to be used by SubItemIt
        typedef Node IterItem;  // needed to be used by SubItemIt

        ParentItFunctor(Graph const * graph)
            : graph_(graph)
        {}

        ParentItFunctor(Graph const & graph)
            : graph_(&graph)
        {}

        void first(Node const & sourcenode, Node & parentnode)
        {
            graph_->firstIn(arc_, sourcenode);
            if (graph_->valid(arc_))
                parentnode = graph_->source(arc_);
            else
                parentnode = lemon::INVALID;
        }

        void next(Node & parentnode)
        {
            graph_->nextIn(arc_);
            if (graph_->valid(arc_))
                parentnode = graph_->source(arc_);
            else
                parentnode = lemon::INVALID;
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
    typedef detail::ItemIt<DAGraph0, detail::NodeItFunctor<DAGraph0> > NodeIt;
    typedef detail::ItemIt<DAGraph0, detail::RootNodeItFunctor<DAGraph0> > RootNodeIt;
    typedef detail::ItemIt<DAGraph0, detail::LeafNodeItFunctor<DAGraph0> > LeafNodeIt;
    typedef detail::ItemIt<DAGraph0, detail::ArcItFunctor<DAGraph0> > ArcIt;
    typedef detail::SubItemIt<DAGraph0, detail::OutArcItFunctor<DAGraph0> > OutArcIt;
    typedef detail::SubItemIt<DAGraph0, detail::InArcItFunctor<DAGraph0> > InArcIt;
    typedef detail::SubItemIt<DAGraph0, detail::ParentItFunctor<DAGraph0> > ParentIt;
    typedef detail::SubItemIt<DAGraph0, detail::ChildItFunctor<DAGraph0> > ChildIt;

    template <typename VALUETYPE>
    struct PropertyMap
    {
        typedef std::map<Node, VALUETYPE> type;
    };
//    template <typename VALUETYPE>
//    using PropertyMap = std::map<Node, VALUETYPE>;

    DAGraph0();

    /// \todo Which of the following are really needed? Are there problems with the defaults?
    DAGraph0(DAGraph0 const &) = default;
    DAGraph0(DAGraph0 &&) = default;
    ~DAGraph0() = default;
    DAGraph0 & operator=(DAGraph0 const &) = default;
    DAGraph0 & operator=(DAGraph0 &&) = default;

    /// \brief Return the maximum of all node ids that were ever used.
    int maxNodeId() const;

    /// \brief Return the maximum of all arc ids that were ever used.
    int maxArcId() const;

    /// \brief Return the source node of the given arc.
    Node source(Arc const & arc) const;

    /// \brief Return the target node of the given arc.
    Node target(Arc const & arc) const;

    /// \brief Set node to the first valid node.
    void first(Node & node) const;

    /// \brief Set node to the next valid node.
    void next(Node & node) const;

    /// \brief Set arc to the first valid arc.
    void first(Arc & arc) const;

    /// \brief Set arc to the next valid arc.
    void next(Arc & arc) const;

    /// \brief Set arc to the first outgoing arc of node.
    void firstOut(Arc & arc, Node const & node) const;

    /// \brief Set arc to the next outgoing arc of node.
    void nextOut(Arc & arc) const;

    /// \brief Set arc to the first incoming arc of node.
    void firstIn(Arc & arc, Node const & node) const;

    /// \brief Set arc to the next incoming arc of node.
    void nextIn(Arc & arc) const;

    /// \brief Return one of the parents of the given node.
    void parent(Node & node) const;

    /// \brief Return one of the children of the given node.
    void child(Node & node) const;

    /// \brief Return the id of the given node.
    static int id(Node const & node);

    /// \brief Return the id of the given arc.
    static int id(Arc const & arc);

    /// \brief Create a node object with the given id.
    static Node nodeFromId(int id);

    /// \brief Create an arc object with the given id.
    static Arc arcFromId(int id);

    /// \brief Return true if the graph contains the given node.
    bool valid(Node const & n) const;

    /// \brief Return true if the graph contains the given arc.
    bool valid(Arc const & a) const;

    /// \brief Add a node to the graph and return it.
    virtual Node addNode();

    /// \brief Add an arc from u to v to the graph and return it.
    virtual Arc addArc(Node const & u, Node const & v);

    /// \brief Remove the given node and all connected arcs from the graph.
    virtual void erase(Node const & node);

    /// \brief Remove the given arc from the graph.
    virtual void erase(Arc const & arc);

    /// \brief Return true if the given node is a root node.
    bool isRootNode(Node const & node) const;

    /// \brief Return true if the given node is a leaf node.
    bool isLeafNode(Node const & node) const;

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

inline auto DAGraph0::source(
        const Arc & arc
) const -> Node
{
    return Node(arcs_[arc.id()].source);
}

inline auto DAGraph0::target(
        const Arc & arc
) const -> Node
{
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

inline void DAGraph0::parent(
        Node & node
) const {
    Arc arc;
    firstIn(arc, node);
    if (valid(arc))
        node = source(arc);
    else
        node = lemon::INVALID;
}

inline void DAGraph0::child(
        Node & node
) const {
    Arc arc;
    firstOut(arc, node);
    if (valid(arc))
        node = target(arc);
    else
        node = lemon::INVALID;
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

inline auto DAGraph0::nodeFromId(
        int id
) -> Node
{
    return Node(id);
}

inline auto DAGraph0::arcFromId(
        int id
) -> Arc
{
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

inline auto DAGraph0::addNode() -> Node
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

inline auto DAGraph0::addArc(
        Node const & u,
        Node const & v
) -> Arc
{
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
    while (valid(arc))
    {
        erase(arc);
        firstOut(arc, node);
    }

    // erase all incoming arcs
    firstIn(arc, node);
    while (valid(arc))
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

inline bool DAGraph0::isRootNode(
        Node const & node
) const {
    Arc tmp;
    firstIn(tmp, node);
    return !valid(tmp);
}

inline bool DAGraph0::isLeafNode(
        Node const & node
) const {
    Arc tmp;
    firstOut(tmp, node);
    return !valid(tmp);
}



/// \brief Hash functor for unordered_set<Node>.
template <class NODE>
struct NodeHash
{
public:
    typedef NODE Node;
    size_t operator()(Node const & node) const
    {
        return h_(node.id());
    }
protected:
    std::hash<int> h_;
};



/// \brief The Forest1 class extends a graph by some rootnode and parent functions. Building Forest1 is slower than building Forest0, but the Forest1 iterators are faster.
template <typename GRAPH>
class Forest1 : public GRAPH
{
public:

    typedef GRAPH Parent;
    typedef typename Parent::Node Node;
    typedef typename Parent::Arc Arc;
    typedef typename Parent::ParentIt ParentIt;
    typedef typename Parent::ChildIt ChildIt;
    typedef std::unordered_set<Node, NodeHash<Node> > ContainerType;
    typedef typename ContainerType::const_iterator const_iterator;
    typedef detail::ItemIt<Forest1, detail::RootNodeVectorItFunctor<Forest1> > RootNodeIt;
    typedef detail::ItemIt<Forest1, detail::LeafNodeVectorItFunctor<Forest1> > LeafNodeIt;

    // TODO: These typedefs are not used within the class. Do they need to be made visible this way? Or are they visible by default (from the superclass)?
    typedef typename Parent::index_type index_type;
    typedef typename Parent::NodeIt NodeIt;
    typedef typename Parent::ArcIt ArcIt;
    typedef typename Parent::OutArcIt OutArcIt;
    typedef typename Parent::InArcIt InArcIt;

    Forest1() = default;
    Forest1(Forest1 const &) = default;
    Forest1(Forest1 &&) = default;
    ~Forest1() = default;
    Forest1 & operator=(Forest1 const &) = default;
    Forest1 & operator=(Forest1 &&) = default;

    /// \brief Construct the forest from the given graph.
    Forest1(Parent const &);

    virtual Node addNode() override;

    virtual Arc addArc(Node const & u, Node const & v) override;

    virtual void erase(Node const & node) override;

    virtual void erase(Arc const & arc) override;

    const_iterator roots_cbegin() const;

    const_iterator roots_cend() const;

    const_iterator leaves_cbegin() const;

    const_iterator leaves_cend() const;

protected:

    /// \brief Unordered set with root nodes.
    ContainerType roots_;

    /// \brief Unordered set with leaf nodes.
    ContainerType leaves_;
};

template <typename GRAPH>
Forest1<GRAPH>::Forest1(Parent const & other)
    : Parent(other)
{
    for (typename Parent::RootNodeIt it(*this); it != lemon::INVALID; ++it)
    {
        roots_.insert(Node(it));
    }
    for (typename Parent::LeafNodeIt it(*this); it != lemon::INVALID; ++it)
    {
        leaves_.insert(Node(it));
    }
}

template <typename GRAPH>
auto Forest1<GRAPH>::addNode() -> Node
{
    Node node = Parent::addNode();
    roots_.insert(node);
    leaves_.insert(node);
    return node;
}

template <typename GRAPH>
auto Forest1<GRAPH>::addArc(
        Node const & u,
        Node const & v
) -> Arc
{
    Arc tmp;
    this->firstOut(tmp, u);
    if (!this->valid(tmp))
        leaves_.erase(u);
    this->firstIn(tmp, v);
    if (!this->valid(tmp))
        roots_.erase(v);

    return Parent::addArc(u, v);
}

template <typename GRAPH>
void Forest1<GRAPH>::erase(
        Node const & node
){
    Parent::erase(node);
    roots_.erase(node);
    leaves_.erase(node);
}

template <typename GRAPH>
void Forest1<GRAPH>::erase(
        Arc const & arc
){
    Node src = this->source(arc);
    Node tar = this->target(arc);
    Parent::erase(arc);

    Node tmp = src;
    this->child(tmp);
    if (!this->valid(tmp))
        leaves_.insert(src);

    tmp = tar;
    this->parent(tmp);
    if (!this->valid(tmp))
        roots_.insert(tar);
}

template <typename GRAPH>
auto Forest1<GRAPH>::roots_cbegin() const -> const_iterator
{
    return roots_.cbegin();
}

template <typename GRAPH>
auto Forest1<GRAPH>::roots_cend() const -> const_iterator
{
    return roots_.cend();
}

template <typename GRAPH>
auto Forest1<GRAPH>::leaves_cbegin() const -> const_iterator
{
    return leaves_.cbegin();
}

template <typename GRAPH>
auto Forest1<GRAPH>::leaves_cend() const -> const_iterator
{
    return leaves_.cend();
}



} // namespace vigra

#endif // VIGRA_DAGRAPH_HXX
