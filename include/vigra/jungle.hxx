#ifndef VIGRA_JUNGLE_HXX
#define VIGRA_JUNGLE_HXX

#include <vigra/graphs.hxx> // for lemon::Invalid

namespace vigra
{



namespace detail
{

    template <typename IDTYPE>
    class Node
    {
    public:
        typedef IDTYPE index_type;
        Node(lemon::Invalid = lemon::INVALID)
            : id_(-1)
        {}
        explicit Node(index_type const & id)
            : id_(id)
        {}
        bool operator!=(Node const & other) const
        {
            return id_ != other.id_;
        }
        bool operator==(Node const & other) const
        {
            return id_ == other.id_;
        }
        bool operator<(Node const & other) const
        {
            return id_ < other.id_;
        }
    protected:
        index_type id_;
    };

    template <typename IDTYPE>
    class Arc
    {
    public:
        typedef IDTYPE index_type;
        Arc(lemon::Invalid = lemon::INVALID)
            : id_(-1)
        {}
        explicit Arc(index_type const & id)
            : id_(id)
        {}
        bool operator!=(Arc const & other) const
        {
            return id_ != other.id_;
        }
        bool operator==(Arc const & other) const
        {
            return id_ == other.id_;
        }
        bool operator<(Arc const & other) const
        {
            return id_ < other.id_;
        }
    protected:
        index_type id_;
    };



}



class ListRandomAccessJungle
{
public:

    typedef Int64 index_type;
    typedef detail::Node<index_type> Node;
    typedef detail::Arc<index_type> Arc;

protected:



};



} // namespace vigra

#endif
