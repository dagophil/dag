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

protected:

    /// \brief Since you cannot add or remove elements, the default constructor will always yield an empty graph, and that's why it is hidden.
    StaticDAGraph()
    {
    }

};



StaticDAGraph StaticDAGraph::build(
        size_t num_nodes,
        std::vector<std::pair<int, int> > const & arcs
){
    StaticDAGraph g;
    g.nodes_.resize(num_nodes, {-1, -1, -1, -1, -1, -1});

    for (auto const & a : arcs)
    {
        if (a.first < 0 || a.first >= num_nodes || a.second < 0 || a.second >= num_nodes)
            throw std::runtime_error("StaticDAGraph::build(): Node index out of range.");

        NodeT & u = g.nodes_[a.first];
        NodeT & v = g.nodes_[a.second];

        int arcid = g.arcs_.size();
        ArcT arc;
        arc.source = a.first;
        arc.target = a.second;
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

        // TODO: Modify root and leaf descriptors.

        g.arcs_.push_back(arc);
    }

    return g;
}



}

#endif // VIGRA_DAGRAPH_HXX
