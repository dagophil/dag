#ifndef VIGRA_DAGRAPH_HXX
#define VIGRA_DAGRAPH_HXX

#include <vector>
#include <utility>
#include <cassert>

#include <vigra/graphs.hxx>

namespace vigra
{



/// \brief Base class for a static directed acyclic graph (static: no nodes or edges can be added/removed).
class StaticDAGraph
{

protected:

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

    /// \brief Construct the graph from a vector of pairs.
    /// \param num_nodes: Number of nodes.
    /// \param arcs: The pairs in this vector give the ids of the nodes that are connected by an arc.
    static StaticDAGraph build(
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
    StaticDAGraph()
        :
          first_root_node_(-1),
          first_leaf_node_(-1)
    {
    }

};



StaticDAGraph StaticDAGraph::build(
        size_t num_nodes,
        std::vector<std::pair<int, int> > const & arcs
){
    StaticDAGraph g;
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
            u.first_out = arcid;
            arc.next_out = -1;
        }
        else
        {
            ArcT & outarc = g.arcs_[u.first_out];
            assert(outarc.prev_out == -1);
            outarc.prev_out = arcid;
            arc.next_out = u.first_out;
        }

        if (v.first_in == -1)
        {
            v.first_in = arcid;
            arc.next_in = -1;
        }
        else
        {
            ArcT & inarc = g.arcs_[v.first_in];
            assert(inarc.prev_in == -1);
            inarc.prev_in = arcid;
            arc.next_in = v.first_in;
        }

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



}

#endif // VIGRA_DAGRAPH_HXX
