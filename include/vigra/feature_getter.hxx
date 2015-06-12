#ifndef VIGRA_FEATURE_GETTER_HXX
#define VIGRA_FEATURE_GETTER_HXX

#include <vigra/multi_array.hxx>
#include <vector>
#include <utility>

namespace vigra
{

namespace detail
{
    /// \brief Proxy class that is used to forward assignments to the SparseFeatureGetter.
    template <typename FEATURES>
    class SparseFeatureGetterProxy
    {
    public:

        typedef FEATURES Features;
        typedef typename Features::value_type value_type;

        SparseFeatureGetterProxy(
                Features & features,
                size_t const i,
                size_t const j
        )   : features_(features),
              i_(i),
              j_(j)
        {}

        operator value_type() const
        {
            if (i_ >= features_.shape()[0] || j_ >= features_.shape()[1])
                vigra_fail("Proxy::operator value_type(): Invalid Proxy.");

            auto const & indices = features_.indices_[i_];
            auto const & values = features_.values_[i_];
            auto const lower = std::lower_bound(indices.begin(), indices.end(), j_);
            if (lower == indices.end() || *lower != j_)
            {
                return 0;
            }
            else
            {
                size_t ind = std::distance(indices.begin(), lower);
                return values[ind];
            }
        }

        SparseFeatureGetterProxy & operator=(value_type const & v)
        {
            if (i_ >= features_.shape()[0] || j_ >= features_.shape()[1])
                vigra_fail("Proxy::operator=(): Invalid Proxy.");

            auto & indices = features_.indices_[i_];
            auto & values = features_.values_[i_];
            auto const lower = std::lower_bound(indices.begin(), indices.end(), j_);
            size_t ind = std::distance(indices.begin(), lower);

            if (v == 0)
            {
                // Delete the value if it was non-zero before.
                if (lower != indices.end() && *lower == j_)
                {
                    indices.erase(lower);
                    values.erase(values.begin()+ind);
                }
            }
            else
            {
                if (lower == indices.end() || *lower != j_)
                {
                    indices.insert(lower, j_);
                    values.insert(values.begin()+ind, v);
                }
                else
                {
                    values[ind] = v;
                }
            }
            return *this;
        }

    protected:

        Features & features_;
        size_t const i_;
        size_t const j_;
    };

    /// \brief Proxy class that is used to forward assignments to the SparseFeatureGetter.
    template <typename FEATURES>
    class SparseFeatureGetterConstProxy
    {
    public:

        typedef FEATURES Features;
        typedef typename Features::value_type value_type;

        SparseFeatureGetterConstProxy(
                Features const & features,
                size_t const i,
                size_t const j
        )   : features_(features),
              i_(i),
              j_(j)
        {}

        operator value_type() const
        {
            if (i_ >= features_.shape()[0] || j_ >= features_.shape()[1])
                vigra_fail("ConstProxy::operator value_type(): Invalid Proxy.");

            auto const & indices = features_.indices_[i_];
            auto const & values = features_.values_[i_];
            auto const lower = std::lower_bound(indices.begin(), indices.end(), j_);
            if (lower == indices.end() || *lower != j_)
            {
                return 0;
            }
            else
            {
                size_t ind = std::distance(indices.begin(), lower);
                return values[ind];
            }
        }

    protected:

        Features const & features_;
        size_t const i_;
        size_t const j_;
    };



    /// \brief Iterator for the non zero elements of a single instance.
    template <typename FEATURES>
    class SparseFeatureGetterConstNonZeroIter
    {
    public:

        typedef FEATURES Features;
        typedef typename Features::value_type value_type;

        SparseFeatureGetterConstNonZeroIter(
                Features const & features,
                size_t const i,
                bool is_end = false
        )   : indices_(features.indices_[i]),
              values_(features.values_[i]),
              current_(0),
              i_(i)
        {
            if (is_end)
            {
                current_ = indices_.size();
            }
        }

        SparseFeatureGetterConstNonZeroIter & operator++()
        {
            ++current_;
        }

        std::pair<size_t, value_type> operator*() const
        {
            return std::pair<size_t, value_type>(indices_[current_], values_[current_]);
        }

        bool operator!=(SparseFeatureGetterConstNonZeroIter const & other)
        {
            return i_ != other.i_ || current_ != other.current_;
        }

    protected:

        std::vector<size_t> const & indices_;
        std::vector<value_type> const & values_;
        size_t current_;
        size_t const i_;

    };

    template <typename FEATURES>
    class SparseFeatureGetterConstIter
    {
    public:

        typedef FEATURES Features;
        typedef typename Features::value_type value_type;

        SparseFeatureGetterConstIter(
                Features const & features,
                size_t const i,
                bool is_end = false
        )   : indices_(features.indices_[i]),
              values_(features.values_[i]),
              current_(0),
              next_index_(0),
              i_(i)
        {
            if (!indices_.empty() && indices_.front() == 0)
            {
                is_zero_ = false;
                ++next_index_;
            }
            else
            {
                is_zero_ = true;
            }
            if (is_end)
            {
                current_ = features.shape()[1];
            }
        }

        SparseFeatureGetterConstIter & operator++()
        {
            ++current_;
            if (current_ == indices_[next_index_])
            {
                is_zero_ = false;
                ++next_index_;
            }
            else
            {
                is_zero_ = true;
            }
        }

        value_type operator*() const
        {
            if (is_zero_)
            {
                return static_cast<value_type>(0);
            }
            else
            {
                return values_[next_index_-1];
            }
        }

        bool operator!=(SparseFeatureGetterConstIter const & other) const
        {
            return i_ != other.i_ || current_ != other.current_;
        }

    protected:

        std::vector<size_t> const & indices_;
        std::vector<value_type> const & values_;
        size_t current_;
        size_t next_index_;
        bool is_zero_;
        size_t const i_;
    };


} // namespace detail



/// \brief Wrapper class for the features. The FeatureGetter saves a reference to a multi array and forwards the calls to that array.
template <typename T>
class FeatureGetter
{
public:
    typedef T value_type;
    typedef value_type & reference;
    typedef value_type const & const_reference;
    typedef typename MultiArrayView<2, T>::difference_type difference_type;

    FeatureGetter(MultiArrayView<2, T> const & arr)
        : arr_(arr)
    {}

    /// \brief Return the features of instance i.
    MultiArrayView<1, T> instance_features(size_t i)
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the features of instance i (const version).
    MultiArrayView<1, T> const instance_features(size_t i) const
    {
        return arr_.template bind<0>(i);
    }

    /// \brief Return the i-th feature of all instances.
    MultiArrayView<1, T> get_features(size_t i)
    {
        return arr_.template bind<1>(i);
    }

    /// \brief Return the i-th feature of all instances (const version).
    MultiArrayView<1, T> const get_features(size_t i) const
    {
        return arr_.template bind<1>(i);
    }

    /// \brief Return the shape of the underlying multi array.
    difference_type const & shape() const
    {
        return arr_.shape();
    }

    /// \brief Return the number of instances.
    size_t num_instances() const
    {
        return arr_.shape()[0];
    }

    /// \brief Return the number of features.
    size_t num_features() const
    {
        return arr_.shape()[1];
    }

    reference operator()(size_t i, size_t j)
    {
        return arr_(i, j);
    }

    const_reference operator()(size_t i, size_t j) const
    {
        return arr_(i, j);
    }

    template <typename ITER>
    void sort(size_t feat, ITER begin, ITER end) const
    {
        auto const & arr = arr_;
        std::sort(begin, end,
                [& arr, & feat](size_t i, size_t j)
                {
                    return arr(i, feat) < arr(j, feat);
                }
        );
    }

protected:
    MultiArrayView<2, T> const & arr_;
};



/// \brief Wrapper class for the features. The SparseFeatureGetter saves sparse data by saving only the non-zero values.
///
/// The SparseFeatureGetter saves two arrays for each instance:
/// One array with the non-zero values and one array with the indices of those values.
template <typename T>
class SparseFeatureGetter
{
public:

    typedef T value_type;
    typedef value_type & reference;
    typedef value_type const & const_reference;
    typedef detail::SparseFeatureGetterProxy<SparseFeatureGetter> Proxy;
    typedef detail::SparseFeatureGetterConstProxy<SparseFeatureGetter> ConstProxy;
    typedef detail::SparseFeatureGetterConstNonZeroIter<SparseFeatureGetter> ConstNonZeroIter;
    typedef detail::SparseFeatureGetterConstIter<SparseFeatureGetter> ConstIter;

    friend Proxy;
    friend ConstProxy;
    friend ConstNonZeroIter;
    friend ConstIter;

    SparseFeatureGetter(Shape2 const & shape = Shape2(0, 0))
        : shape_(shape),
          indices_(shape[0]),
          values_(shape[0])
    {}

    SparseFeatureGetter(MultiArrayView<2, value_type> const & arr)
        : shape_(arr.shape()),
          indices_(arr.shape()[0]),
          values_(arr.shape()[0])
    {
        for (size_t i = 0; i < arr.shape()[0]; ++i)
        {
            for (size_t j = 0; j < arr.shape()[1]; ++j)
            {
                if (arr(i, j) != 0)
                {
                    indices_[i].push_back(j);
                    values_[i].push_back(arr(i, j));
                }
            }
        }
    }

    void reshape(Shape2 const & shape)
    {
        shape_ = shape;
        indices_.resize(shape_[0]);
        values_.resize(shape_[0]);
    }

    Shape2 const & shape() const
    {
        return shape_;
    }

    size_t const size() const
    {
        return shape_[0]*shape_[1];
    }

    Proxy operator()(size_t const i, size_t const j)
    {
        return Proxy(*this, i, j);
    }

    ConstProxy const operator()(size_t const i, size_t const j) const
    {
        return ConstProxy(*this, i, j);
    }

    size_t count_nonzero() const
    {
        size_t count = 0;
        for (auto const & v : indices_)
        {
            count += v.size();
        }
        return count;
    }

    ConstIter begin_instance(size_t const i) const
    {
        return ConstIter(*this, i);
    }

    ConstIter end_instance(size_t const i) const
    {
        return ConstIter(*this, i, true);
    }

    ConstNonZeroIter begin_instance_nonzero(size_t const i) const
    {
        return ConstNonZeroIter(*this, i);
    }

    ConstNonZeroIter end_instance_nonzero(size_t const i) const
    {
        return ConstNonZeroIter(*this, i, true);
    }

    /// \brief Insert the value v for feature j of instance i at the end of the internal storage vectors.
    ///
    /// \note Precondition: There must not be a non-zero value at (i, k) for all k >= j.
    void unsafe_insert(size_t const i, size_t const j, value_type const & v)
    {
        if (!indices_[i].empty() && indices_[i].back() >= j)
            throw std::runtime_error("SparseFeatureGetter::unsafe_insert(): The precondition is not fulfilled.");

        if (v != 0)
        {
            indices_[i].push_back(j);
            values_[i].push_back(v);
        }
    }

protected:

    std::vector<std::vector<size_t> > indices_;
    std::vector<std::vector<value_type> > values_;
    Shape2 shape_;
};



template <typename T>
class LabelGetter
{
public:
    typedef T value_type;
    typedef value_type & reference;
    typedef value_type const & const_reference;
    typedef typename MultiArrayView<1, T>::iterator iterator;
    typedef typename MultiArrayView<1, T>::const_iterator const_iterator;
    typedef typename MultiArrayView<1, T>::difference_type difference_type;

    LabelGetter(MultiArrayView<1, T> const & arr)
        : arr_(arr)
    {}

    /// \brief Return the label for instance i.
    reference operator()(size_t i)
    {
        return arr_(i);
    }

    /// \brief Return the label for instance i.
    const_reference operator()(size_t i) const
    {
        return arr_(i);
    }

    /// \brief Return the number of instances.
    size_t size() const
    {
        return arr_.size();
    }

    /// \brief Return the number of instances.
    size_t num_instances() const
    {
        return arr_.size();
    }

    iterator begin()
    {
        return arr_.begin();
    }

    const_iterator begin() const
    {
        return arr_.begin();
    }

    iterator end()
    {
        return arr_.end();
    }

    const_iterator end() const
    {
        return arr_.end();
    }

    difference_type const & shape() const
    {
        return arr_.shape();
    }

protected:
    MultiArrayView<1, value_type> const & arr_;
};



} // namespace vigra

#endif
