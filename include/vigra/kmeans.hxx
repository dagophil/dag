#ifndef VIGRA_KMEANS_HXX
#define VIGRA_KMEANS_HXX

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <type_traits>

namespace vigra
{



namespace detail
{
    template <typename ARRAY>
    class LineView1DIter
    {
    public:

        typedef ARRAY Array;
        typedef typename Array::value_type value_type;

        LineView1DIter(
                Array const & arr,
                std::vector<size_t> const & lines
        )   : arr_(arr),
              lines_(lines),
              i_(0)
        {}

        LineView1DIter & operator++()
        {
            ++i_;
        }

        value_type const & operator*() const
        {
            return arr_(lines_[i_]);
        }

        /// \note We already assume that the iterators both point to the same array.
        bool operator!=(LineView1DIter const & other) const
        {
            return i_ != other.i_;
        }

        /// \brief Make this iterator become the end iterator.
        void make_end()
        {
            i_ = lines_.size();
        }

    protected:

        Array const & arr_;
        std::vector<size_t> const & lines_;
        size_t i_;
    };

    template <typename ARRAY>
    class LineView2DIter
    {
    public:

        typedef ARRAY Array;
        typedef typename Array::value_type value_type;

        LineView2DIter(
                Array const & arr,
                std::vector<size_t> const & lines
        )   : arr_(arr),
              lines_(lines),
              i_(0),
              j_(0)
        {}

        LineView2DIter & operator++()
        {
            ++i_;
            if (i_ == lines_.size())
            {
                i_ = 0;
                ++j_;
            }
        }

        value_type const & operator*() const
        {
            return arr_(lines_[i_], j_);
        }

        bool operator!=(LineView2DIter const & other) const
        {
            return i_ != other.i_ || j_ != other.j_;
        }

        /// \brief Make this iterator become the end iterator.
        void make_end()
        {
            i_ = 0;
            j_ = arr_.shape()[1];
        }


    protected:

        Array const & arr_;
        std::vector<size_t> const & lines_;
        size_t i_;
        size_t j_;

    };


    /// \brief Class that acts like a multi array view, but restricts the access to the first coordinate to the given lines.
    ///
    /// Only implemented for 1d or 2d arrays.
    template <typename ARRAY>
    class LineView
    {
    public:

        typedef ARRAY Array;
        typedef typename Array::value_type value_type;
        typedef typename Array::difference_type difference_type;
        typedef typename std::conditional<std::is_same<difference_type, Shape1>::value, LineView1DIter<Array>, LineView2DIter<Array> >::type Iter;

        LineView(
                Array const & arr,
                std::vector<size_t> const & lines
        )   : arr_(arr),
              lines_(lines),
              shape_(arr.shape())
        {
            static_assert(std::is_same<difference_type, Shape1>() || std::is_same<difference_type, Shape2>(),
                          "LineView(): Only implemented for 1 or 2 dimensional arrays.");
            shape_[0] = lines_.size();
        }

        difference_type const & shape() const
        {
            return shape_;
        }

        value_type const & operator()(size_t i) const
        {
            return arr_(lines_[i]);
        }

        value_type const & operator()(size_t i, size_t j) const
        {
            return arr_(lines_[i], j);
        }

        size_t  size() const
        {
            return (arr_.size() / arr_.shape()[0]) * shape_[0];
        }

        Iter begin() const
        {
            return Iter(arr_, lines_);
        }

        Iter end() const
        {
            Iter enditer(arr_, lines_);
            enditer.make_end();
            return enditer;
        }

    protected:

        Array const & arr_;
        std::vector<size_t> const & lines_;
        difference_type shape_;
    };
}



struct KMeansStoppingCriteria
{
    explicit KMeansStoppingCriteria(
            double const min_dist_improvement = 0.01,
            size_t max_t = std::numeric_limits<size_t>::max()
    )   : min_dist_improvement_(min_dist_improvement),
          max_t_(max_t)
    {}

    // Stop if old_sum_of_distances / new_sum_of_distances < 1+min_dist_improvement for at least 3 times.
    double min_dist_improvement_;

    // Stop after max_t interations.
    size_t max_t_;
};



/// \brief Run kmeans algorithm on points to find k clusters.
///
/// \param points: the points
/// \param k: number of clusters
/// \param instance_clusters[out]: the cluster id of each instance
/// \param stop: the stopping criteria
/// \param considered_instances: only consider these instances when computing the clusters (if empty: take all instances)
/// \param randengine: the random engine
template <typename FEATURES, typename RANDENGINE = MersenneTwister>
void kmeans(
        FEATURES const & points,
        size_t const k,
        std::vector<size_t> & instance_clusters,
        KMeansStoppingCriteria const & stop = KMeansStoppingCriteria(),
        std::vector<size_t> considered_instances = std::vector<size_t>(),
        RANDENGINE const & randengine = RANDENGINE::global()
){
    vigra_precondition(k > 0, "kmeans(): k must not be negative.");

    // Create the view on the considered instances.
    bool reassign = true;
    if (considered_instances.empty())
    {
        reassign = false;
        considered_instances.resize(points.shape()[0]);
        std::iota(considered_instances.begin(), considered_instances.end(), 0);
    }
    detail::LineView<FEATURES> const points_sub(points, considered_instances);
    size_t const num_instances = points_sub.shape()[0];
    size_t const num_features = points_sub.shape()[1];

    // Find the initial clustering (use the instance_clusters vector, so no additional space is required).
    instance_clusters.resize(num_instances);
    MultiArray<2, double> clusters(Shape2(k, num_features));
    UniformIntRandomFunctor<MersenneTwister> rand(randengine);
    std::iota(instance_clusters.begin(), instance_clusters.end(), 0);
    for (size_t c = 0; c < k; ++c)
    {
        size_t const ii = c+rand(num_instances-c);
        std::swap(instance_clusters[c], instance_clusters[ii]);
        for (size_t j = 0; j < num_features; ++j)
        {
            clusters(c, j) = points_sub(instance_clusters[c], j);
        }
    }

    // Counter for the number of instances in each cluster.
    std::vector<size_t> instance_count(k);

    // Do the kmeans loop.
    double sum_of_distances = std::numeric_limits<double>::max();
    size_t sum_violated_counter = 0;
    for (size_t t = 0; t < stop.max_t_; ++t)
    {
        // Reset the counter.
        for (size_t c = 0; c < k; ++c)
        {
            instance_count[c] = 0;
        }
        double old_sum_of_distances = sum_of_distances;
        sum_of_distances = 0.;

        // Assign each instance to its cluster.
        for (size_t i = 0; i < num_instances; ++i)
        {
            size_t best_cluster = 0;
            double best_distance = std::numeric_limits<double>::max();
            for (size_t c = 0; c < k; ++c)
            {
                double distance = 0.;
                for (size_t j = 0; j < num_features; ++j)
                {
                    double v = points_sub(i, j) - clusters(c, j);
                    distance += v*v;
                }
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_cluster = c;
                }
            }
            instance_clusters[i] = best_cluster;
            ++instance_count[best_cluster];
            sum_of_distances += best_distance;
        }

        // Check the sum-of-distances stopping criterion.
        if (old_sum_of_distances < sum_of_distances * (1 + stop.min_dist_improvement_))
        {
            ++sum_violated_counter;
        }
        else
        {
            sum_violated_counter = 0;
        }
        if (sum_violated_counter >= 3)
        {
            break;
        }

        // Compute the new cluster centers.
        clusters = 0.;
        for (size_t i = 0; i < num_instances; ++i)
        {
            size_t const c = instance_clusters[i];
            for (size_t j = 0; j < num_features; ++j)
            {
                clusters(c, j) += points_sub(i, j);
            }
        }
        for (size_t c = 0; c < k; ++c)
        {
            for (size_t j = 0; j < num_features; ++j)
            {
                clusters(c, j) /= static_cast<double>(instance_count[c]);
            }
        }
    }

    // If only a subsample of the instances was used to find the cluster centers, assign the clusters to the whole set of instances.
    if (reassign)
    {
        instance_clusters.resize(points.shape()[0]);
        for (size_t i = 0; i < points.shape()[0]; ++i)
        {
            double best_distance = std::numeric_limits<double>::max();
            for (size_t c = 0; c < k; ++c)
            {
                double distance = 0.;
                for (size_t j = 0; j < num_features; ++j)
                {
                    double v = points(i, j) - clusters(c, j);
                    distance += v*v;
                }
                if (distance < best_distance)
                {
                    best_distance = distance;
                    instance_clusters[i] = c;
                }
            }
        }
    }
}



} // namespace vigra


#endif
