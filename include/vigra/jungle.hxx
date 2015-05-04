#ifndef VIGRA_JUNGLE_HXX
#define VIGRA_JUNGLE_HXX

#include <vigra/graphs.hxx> // for lemon::Invalid
#include <vector>
#include <utility>
#include <algorithm>

namespace vigra
{



namespace detail
{

    template <typename IDTYPE>
    class IndexNode
    {
    public:
        typedef IDTYPE index_type;
        IndexNode(lemon::Invalid = lemon::INVALID)
            : id_(-1)
        {}
        explicit IndexNode(index_type const & id)
            : id_(id)
        {}
        bool operator!=(IndexNode const & other) const
        {
            return id_ != other.id_;
        }
        bool operator==(IndexNode const & other) const
        {
            return id_ == other.id_;
        }
        bool operator<(IndexNode const & other) const
        {
            return id_ < other.id_;
        }
        index_type id() const
        {
            return id_;
        }
    protected:
        index_type id_;
    };

    template <typename IDTYPE>
    class IndexArc
    {
    public:
        typedef IDTYPE index_type;
        IndexArc(lemon::Invalid = lemon::INVALID)
            : id_(-1)
        {}
        explicit IndexArc(index_type const & id)
            : id_(id)
        {}
        bool operator!=(IndexArc const & other) const
        {
            return id_ != other.id_;
        }
        bool operator==(IndexArc const & other) const
        {
            return id_ == other.id_;
        }
        bool operator<(IndexArc const & other) const
        {
            return id_ < other.id_;
        }
        index_type id() const
        {
            return id_;
        }
    protected:
        index_type id_;
    };

    template <typename S, typename T>
    class InvPair : public std::pair<S, T>
    {
    public:
        typedef std::pair<S, T> Parent;
        explicit InvPair(int i)
            : Parent(i, i)
        {}
        InvPair(S const & a, T const & b)
            : Parent(a, b)
        {}
    };

}



class BinaryTree
{
public:

    typedef Int64 index_type;
    typedef detail::IndexNode<index_type> Node;
    typedef detail::IndexArc<index_type> Arc;

    // IterDAG API

    BinaryTree();

    Node addNode();

    Arc addArc(Node const & u, Node const & v);

    void erase(Node const & node);

    void erase(Arc const & arc);

    Node source(Arc const & arc) const;

    Node target(Arc const & arc) const;

    index_type id(Node const & node) const;

    index_type id(Arc const & arc) const;

    Node nodeFromId(index_type const & id) const;

    Arc arcFromId(index_type const & id) const;

    size_t numNodes() const;

    size_t numArcs() const;

    size_t inDegree(Node const & node) const;

    size_t outDegree(Node const & node) const;

    size_t numParents(Node const & node) const;

    size_t numChildren(Node const & node) const;

    // RandomAccessDAG API

    Arc getOutArc(Node const & node, size_t i) const;

    Arc getInArc(Node const & node, size_t i) const;

    Node getParent(Node const & node, size_t i) const;

    Node getChild(Node const & node, size_t i) const;

    // SingleRootIterDAG API

    size_t numLeaves() const;

    Node getRoot() const;

    // SingleRootRandomAccess API

    Node getLeafNode(size_t i) const;

protected:

    struct NodeT
    {
        index_type prev;
        index_type next;
        index_type parent;
        index_type left_child;
        index_type right_child;
    };

    // arc_id = 2*source_node_id + x
    // x = 0 if arc is out arc to left child
    // x = 1 if arc is out arc to right child

    std::vector<NodeT> nodes_;
    index_type first_node_;
    index_type first_free_node_;

};

inline BinaryTree::BinaryTree()
    : nodes_(),
      first_node_(-1),
      first_free_node_(-1)
{}

inline BinaryTree::Node BinaryTree::addNode()
{
    index_type id;

    if (first_free_node_ == -1)
    {
        id = nodes_.size();
        nodes_.push_back(NodeT());
    }
    else
    {
        id = first_free_node_;
        first_free_node_ = nodes_[id].next;
    }

    nodes_[id].next = first_node_;
    if (first_node_ != -1)
        nodes_[first_node_].prev = id;
    first_node_ = id;
    nodes_[id].prev = -1;
    nodes_[id].parent = -1;
    nodes_[id].left_child = -1;
    nodes_[id].right_child = -1;

    return Node(id);
}

inline BinaryTree::Arc BinaryTree::addArc(Node const & u, Node const & v)
{
    vigra_precondition(nodes_[v.id()].parent == -1,
            "BinaryTree::addArc(): The node v already has a parent.");

    NodeT & n = nodes_[u.id()];
    if (n.left_child == -1)
    {
        n.left_child = v.id();
    }
    else if (n.right_child == -1)
    {
        n.right_child = v.id();
    }
    else
    {
        vigra_fail("BinaryTree::addArc(): The node u already has two children.");
    }
    nodes_[v.id()].parent = u.id();
}

inline void BinaryTree::erase(Node const & node)
{
    index_type id = node.id();
    NodeT & n = nodes_[id];

    // erase the node as parent of its children
    if (n.left_child != -1)
    {
        nodes_[n.left_child].parent = -1;
    }
    if (n.right_child != -1)
    {
        nodes_[n.right_child].parent = -1;
    }

    // erase the node as child of its parent
    if (n.parent != -1)
    {
        if (nodes_[n.parent].left_child == id)
        {
            nodes_[n.parent].left_child = -1;
        }
        else if (nodes_[n.parent].right_child == id)
        {
            nodes_[n.parent].right_child = -1;
        }
        else
        {
            vigra_fail("BinaryTree::erase(): The node is not registered as child of its parent.");
        }
    }

    // erase the node itself
    if (n.next != -1)
    {
        nodes_[n.next].prev = n.prev;
    }
    if (n.prev != -1)
    {
        nodes_[n.prev].next = n.next;
    }
    else
    {
        first_node_ = n.next;
    }

    n.next = first_free_node_;
    first_free_node_ = id;
    n.prev = -2;
}

inline void BinaryTree::erase(Arc const & arc)
{
    index_type const id = arc.id();
    NodeT & src = nodes_[id/2];

    if (id % 2 == 0)
    {
        nodes_[src.left_child].parent = -1;
        src.left_child = -1;
    }
    else
    {
        nodes_[src.right_child].parent = -1;
        src.right_child = -1;
    }
}

inline BinaryTree::Node BinaryTree::source(Arc const & arc) const
{
    return Node(arc.id()/2);
}

inline BinaryTree::Node BinaryTree::target(Arc const & arc) const
{
    index_type id = arc.id();
    NodeT const & src = nodes_[id/2];
    if (id % 2 == 0)
    {
        return Node(src.left_child);
    }
    else
    {
        return Node(src.right_child);
    }
}

inline BinaryTree::index_type BinaryTree::id(Node const & node) const
{
    return node.id();
}

inline BinaryTree::index_type BinaryTree::id(Arc const & arc) const
{
    return arc.id();
}

inline BinaryTree::Node BinaryTree::nodeFromId(index_type const & id) const
{
    return Node(id);
}

inline BinaryTree::Arc BinaryTree::arcFromId(index_type const & id) const
{
    return Arc(id);
}

inline size_t BinaryTree::numNodes() const
{
    // TODO: Implement this.
    vigra_fail("BinaryTree::numNodes(): Not implemented yet.");
}

inline size_t BinaryTree::numArcs() const
{
    // TODO: Implement this.
    vigra_fail("BinaryTree::numArcs(): Not implemented yet.");
}

inline size_t BinaryTree::inDegree(Node const & node) const
{
    if (nodes_[node.id()].parent == -1)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

inline size_t BinaryTree::outDegree(Node const & node) const
{
    NodeT const & n = nodes_[node.id()];
    if (n.left_child == -1 && n.right_child == -1)
    {
        return 0;
    }
    else if (n.left_child == -1)
    {
        return 1;
    }
    else if (n.right_child == -1)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

inline size_t BinaryTree::numParents(Node const & node) const
{
    return inDegree(node);
}

inline size_t BinaryTree::numChildren(Node const & node) const
{
    return outDegree(node);
}

inline BinaryTree::Arc BinaryTree::getOutArc(Node const & node, size_t i) const
{
    vigra_precondition(i == 0 || i == 1, "BinaryTree::getOutArc(): Index out of range.");
    return Arc(2*node.id()+i);
}

inline BinaryTree::Arc BinaryTree::getInArc(Node const & node, size_t i) const
{
    vigra_precondition(i == 0, "BinaryTree::getInArc(): Index out of range.");
    index_type const p_id = nodes_[node.id()].parent;
    NodeT const & p = nodes_[p_id];
    if (p.left_child == node.id())
    {
        return Arc(2*p_id);
    }
    else
    {
        return Arc(2*p_id + 1);
    }
}

inline BinaryTree::Node BinaryTree::getParent(Node const & node, size_t i) const
{
    vigra_precondition(i == 0, "BinaryTree::getParent(): Index out of range.");
    return Node(nodes_[node.id()].parent);
}

inline BinaryTree::Node BinaryTree::getChild(Node const & node, size_t i) const
{
    if (i == 0)
    {
        return Node(nodes_[node.id()].left_child);
    }
    else if (i == 1)
    {
        return Node(nodes_[node.id()].right_child);
    }
    else
    {
        vigra_fail("BinaryTree::getChild(): Index out of range.");
    }
}

inline size_t BinaryTree::numLeaves() const
{
    // TODO: Implement.
    vigra_fail("Not implemented yet.");
}

inline BinaryTree::Node BinaryTree::getRoot() const
{
    // TODO: Implement.
    vigra_fail("Not implemented yet.");
}

inline BinaryTree::Node BinaryTree::getLeafNode(size_t i) const
{
    // TODO: Implement.
    vigra_fail("Not implemented yet.");
}



//template <typename TREE>
//class RandomAccessForest0
//{
//public:

//    typedef TREE Tree;

//    RandomAccessForest0();

//protected:

//    std::vector<Tree> trees_;

//};

//template <typename TREE>
//RandomAccessForest0<TREE>::RandomAccessForest0()
//    : trees_(10)
//{}



} // namespace vigra

#endif
