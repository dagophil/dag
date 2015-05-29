#ifndef VIGRA_JUNGLE_HXX
#define VIGRA_JUNGLE_HXX

#include <vigra/graphs.hxx> // for lemon::Invalid
#include <vector>
#include <utility>
#include <algorithm>
#include <map>

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
    std::ostream & operator << (std::ostream & os, IndexNode<IDTYPE> const & item)
    {
        return os << item.id();
    }

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

    template <typename IDTYPE>
    std::ostream & operator << (std::ostream & os, IndexArc<IDTYPE> const & item)
    {
        return os << item.id();
    }

    template <typename KEYTYPE, typename VALUETYPE, typename MAP = std::map<KEYTYPE, VALUETYPE> >
    class PropertyMap
    {
    public:
        typedef KEYTYPE key_type;
        typedef VALUETYPE value_type;
        typedef value_type & reference;
        typedef value_type const & const_reference;
        typedef MAP Map;
        typedef typename Map::iterator iterator;
        typedef typename Map::const_iterator const_iterator;
        reference at(key_type const & k)
        {
            return map_.at(k);
        }
        const_reference at(key_type const & k) const
        {
            return map_.at(k);
        }
        reference operator[](key_type const & k)
        {
            return map_[k];
        }
        iterator begin()
        {
            return map_.begin();
        }
        const_iterator begin() const
        {
            return map_.begin();
        }
        iterator end()
        {
            return map_.end();
        }
        const_iterator end() const
        {
            return map_.end();
        }
        void clear()
        {
            map_.clear();
        }

    protected:
        Map map_;
    };

}



/// \todo: Change to n-tree (so binary tree is a 2-tree).
class BinaryTree
{
public:

    typedef Int64 index_type;
    typedef detail::IndexNode<index_type> Node;
    typedef detail::IndexArc<index_type> Arc;

    template <typename T>
    using NodeMap = detail::PropertyMap<Node, T>;

    // IterDAG API

    BinaryTree();

    Node addNode();

    Arc addArc(Node const & u, Node const & v);

    void erase(Node const & node);

    void erase(Arc const & arc);

    bool valid(Node const & node) const;

    bool valid(Arc const & arc) const;

    Node source(Arc const & arc) const;

    Node target(Arc const & arc) const;

    index_type id(Node const & node) const;

    index_type id(Arc const & arc) const;

    Node nodeFromId(index_type const & id) const;

    Arc arcFromId(index_type const & id) const;

    index_type maxNodeId() const;

    size_t numNodes() const;

    size_t numArcs() const;

    size_t inDegree(Node const & node) const;

    size_t outDegree(Node const & node) const;

    size_t numParents(Node const & node) const;

    size_t numChildren(Node const & node) const;

    // RandomAccessDAG API

    Arc getOutArc(Node const & node, size_t i) const;

    Arc getInArc(Node const & node, size_t i) const;

    Node getParent(Node const & node, size_t i = 0) const;

    Node getChild(Node const & node, size_t i) const;

    // SingleRootIterDAG API

    size_t numLeaves() const;

    Node getRoot() const;

    // SingleRootRandomAccess API

    Node getLeafNode(size_t i) const;

    /// \brief Return the index i that gives the node with getLeafNode(i).
    size_t getLeafIndex(Node const & node) const;

protected:

    void makeLeaves() const;

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
    size_t num_nodes_;
    size_t num_arcs_;
    mutable index_type root_node_;
    mutable bool root_changed_;
    mutable std::vector<size_t> leaf_ids_; // vector with the node ids of all leaf nodes
    mutable bool leaves_changed_;
    mutable NodeMap<size_t> leaf_indices_; // leaf node map that stores the index in the leaf_ids_ vector

};

inline BinaryTree::BinaryTree()
    : nodes_(),
      first_node_(-1),
      first_free_node_(-1),
      num_nodes_(0),
      num_arcs_(0),
      root_node_(0),
      root_changed_(false),
      leaf_ids_(),
      leaves_changed_(true),
      leaf_indices_()
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

    ++num_nodes_;
    leaves_changed_ = true;
    return Node(id);
}

inline BinaryTree::Arc BinaryTree::addArc(
        Node const & u,
        Node const & v
){
    vigra_precondition(nodes_[v.id()].parent == -1,
            "BinaryTree::addArc(): The node v already has a parent.");

    index_type arc_id = 2*u.id();
    NodeT & n = nodes_[u.id()];
    if (n.left_child == -1)
    {
        n.left_child = v.id();
    }
    else if (n.right_child == -1)
    {
        n.right_child = v.id();
        ++arc_id;
    }
    else
    {
        vigra_fail("BinaryTree::addArc(): The node u already has two children.");
    }

    nodes_[v.id()].parent = u.id();

    root_changed_ = true;
    leaves_changed_ = true;
    ++num_arcs_;
    return Arc(arc_id);
}

inline void BinaryTree::erase(
        Node const & node
){
    index_type id = node.id();
    NodeT & n = nodes_[id];

    // Erase the node as parent of its children.
    if (n.left_child != -1)
    {
        nodes_[n.left_child].parent = -1;
        --num_arcs_;
    }
    if (n.right_child != -1)
    {
        nodes_[n.right_child].parent = -1;
        --num_arcs_;
    }

    // Erase the node as child of its parent.
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
        --num_arcs_;
    }

    // Erase the node itself.
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
    --num_nodes_;
    root_changed_ = true;
    leaves_changed_ = true;
}

inline void BinaryTree::erase(
        Arc const & arc
){
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
    --num_arcs_;
    root_changed_ = true;
    leaves_changed_ = true;
}

inline bool BinaryTree::valid(
        Node const & node
) const {
    return node.id() >= 0 && node.id() < nodes_.size() && nodes_[node.id()].prev != -2;
}

inline bool BinaryTree::valid(
        Arc const & arc
) const {
    index_type const id = arc.id();
    index_type const node_id = id/2;
    if (!valid(Node(node_id)))
        return false;
    return (nodes_[node_id].left_child != -1 && id % 2 == 0) || (nodes_[node_id].right_child != -1 && id % 2 == 1);
}

inline BinaryTree::Node BinaryTree::source(
        Arc const & arc
) const {
    return Node(arc.id()/2);
}

inline BinaryTree::Node BinaryTree::target(
        Arc const & arc
) const {
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

inline BinaryTree::index_type BinaryTree::id(
        Node const & node
) const {
    return node.id();
}

inline BinaryTree::index_type BinaryTree::id(
        Arc const & arc
) const {
    return arc.id();
}

inline BinaryTree::Node BinaryTree::nodeFromId(
        index_type const & id
) const {
    return Node(id);
}

inline BinaryTree::Arc BinaryTree::arcFromId(
        index_type const & id
) const {
    return Arc(id);
}

inline BinaryTree::index_type BinaryTree::maxNodeId() const
{
    return nodes_.size()-1;
}

inline size_t BinaryTree::numNodes() const
{
    return num_nodes_;
}

inline size_t BinaryTree::numArcs() const
{
    return num_arcs_;
}

inline size_t BinaryTree::inDegree(
        Node const & node
) const {
    if (nodes_[node.id()].parent == -1)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

inline size_t BinaryTree::outDegree(
        Node const & node
) const {
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

inline size_t BinaryTree::numParents(
        Node const & node
) const {
    return inDegree(node);
}

inline size_t BinaryTree::numChildren(
        Node const & node
) const {
    return outDegree(node);
}

inline BinaryTree::Arc BinaryTree::getOutArc(
        Node const & node,
        size_t i
) const {
    vigra_precondition(i == 0 || i == 1, "BinaryTree::getOutArc(): Index out of range.");
    return Arc(2*node.id()+i);
}

inline BinaryTree::Arc BinaryTree::getInArc(
        Node const & node,
        size_t i
) const {
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

inline BinaryTree::Node BinaryTree::getParent(
        Node const & node,
        size_t i
) const {
    vigra_precondition(i == 0, "BinaryTree::getParent(): Index out of range.");
    return Node(nodes_[node.id()].parent);
}

inline BinaryTree::Node BinaryTree::getChild(
        Node const & node,
        size_t i
) const {
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
    makeLeaves();
    return leaf_ids_.size();
}

inline BinaryTree::Node BinaryTree::getRoot() const
{
    if (root_changed_)
    {
        index_type current = first_node_;
        while (current != -1)
        {
            if (nodes_[current].parent == -1)
            {
                root_changed_ = false;
                root_node_ = current;
                return Node(root_node_);
            }
            current = nodes_[current].next;
        }
        vigra_fail("BinaryTree::getRoot(): The tree has no root.");
    }
    else
    {
        return Node(root_node_);
    }
}

inline BinaryTree::Node BinaryTree::getLeafNode(
        size_t i
) const {
    makeLeaves();
    return Node(leaf_ids_[i]);
}

inline void BinaryTree::makeLeaves() const
{
    if (leaves_changed_)
    {
        leaf_ids_.clear();
        leaf_indices_.clear();
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            if (outDegree(Node(i)) == 0)
            {
                leaf_ids_.push_back(i);
                leaf_indices_[Node(i)] = leaf_ids_.size()-1;
            }
        }
    }
    leaves_changed_ = false;
}

inline size_t BinaryTree::getLeafIndex(
        Node const & node
) const {
    makeLeaves();
//    return leaf_indices_[node];
    return leaf_indices_.at(node);
}



/// \brief Takes a vector of trees and provides the graph api.
template <typename TREE>
class ConstForestAdaptor
{
public:

    typedef TREE Tree;
    typedef Int64 index_type;
    typedef detail::IndexNode<index_type> Node;
    typedef typename Tree::Node TreeNode;
    // TODO: Implement the rest of the graph API.

    template <typename T>
    using NodeMap = detail::PropertyMap<Node, T>;

    ConstForestAdaptor()
        : forest_()
    {}

    ConstForestAdaptor(std::vector<Tree> const & forest)
        : forest_(forest)
    {}

    /// \brief Setter for forest.
    void set_forest(std::vector<Tree> const & forest)
    {
        forest_ = forest;
    }

    /// \brief Return a node descriptor for the given node in the given tree.
    Node tree_to_forest(
            size_t tree_index,
            TreeNode const & tree_node
    ) const {
        size_t id = forest_[tree_index].getLeafIndex(tree_node);
        for (size_t i = 0; i < tree_index; ++i)
        {
            id += forest_[i].numLeaves();
        }
        return Node(id);
    }

    /// \brief Take the give node, find the tree that holds it and return tree index and the tree node.
    void forest_to_tree(
            Node const & node,
            size_t & tree_index,
            TreeNode & tree_node
    ) const {
        size_t id = node.id();
        for (tree_index = 0; tree_index < forest_.size(); ++tree_index)
        {
            if (id < forest_[tree_index].numLeaves())
            {
                tree_node = TreeNode(forest_[tree_index].getLeafNode(id));
                return;
            }
            else
            {
                id -= forest_[tree_index].numLeaves();
            }
        }
        vigra_fail("ConstForestAdaptor::forest_to_tree(): The given node is not in the graph.");
    }

    /// \brief Return the number of leaves.
    size_t numLeaves() const
    {
        size_t num = 0;
        for (auto const & tree : forest_)
        {
            num += tree.numLeaves();
        }
        return num;
    }

    /// \brief Return the i-th leaf node in the forest.
    Node getLeafNode(size_t index) const
    {
        return Node(index);
    }

    /// \brief Return the index of the given node.
    size_t getLeafIndex(Node const & node) const
    {
        return node.id();
    }

    /// \brief Return a single node map that contains the data of the given tree node maps.
    template <typename TREEMAP>
    NodeMap<typename TREEMAP::value_type> merge_node_maps(
            std::vector<TREEMAP> const & maps
    ) const {
        return merge_maps<NodeMap<typename TREEMAP::value_type>, TREEMAP>(maps);
    }

    /// \brief Return a single property map that contains the data of the given tree property maps.
    template <typename MAP, typename TREEMAP>
    MAP merge_maps(
            std::vector<TREEMAP> const & maps
    ) const {
        vigra_precondition(maps.size() == forest_.size(),
                           "ConstForestAdaptor::merge_maps(): Number of maps must be equal to number of trees.");

        MAP merged_map;
        for (size_t i = 0; i < maps.size(); ++i)
        {
            TREEMAP const & map = maps[i];
            for (auto it = map.begin(); it != map.end(); ++it)
            {
                Node forest_node = tree_to_forest(i, it->first);
                merged_map[forest_node] = it->second;
            }
        }
        return merged_map;
    }

protected:

    std::vector<Tree> forest_;
};



} // namespace vigra

#endif
