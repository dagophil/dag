#ifndef VIGRA_SVM_HXX
#define VIGRA_SVM_HXX

#include <vector>
#include <set>
#include <map>

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>

#include <vigra/kmeans.hxx>



namespace vigra
{



/// \brief Two class support vector machine using hinge loss and a quadratic regularizer (implemented with "dual coordinate descent" [Hsieh et al. 2008]).
template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE = MersenneTwister>
class TwoClassSVM
{
public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef RANDENGINE RandEngine;

    struct StoppingCriteria
    {
        explicit StoppingCriteria(
                double const alpha_tol = 0.0001,
                size_t const max_total_diffs = 0,
                double const max_relative_diffs = 0.,
                double const grad_tol = 0.0001,
                size_t const max_t = std::numeric_limits<size_t>::max()
        )   : alpha_tol_(alpha_tol),
              max_total_diffs_(max_total_diffs),
              max_relative_diffs_(max_relative_diffs),
              grad_tol_(grad_tol),
              max_t_(max_t)
        {
            vigra_precondition(0. <= max_relative_diffs_ && max_relative_diffs_ <= 1.,
                               "StoppingCriteria(): The relative number of differences must be in the interval [0, 1].");
        }

        // count the number of alpha updates that were greater than (or equal to) alpha_tol and
        // stop if count <= max_total_diffs or count <= num_instances*max_relative_diffs
        double alpha_tol_ = 0.0001;
        size_t max_total_diffs_ = 0;
        double max_relative_diffs_ = 0.;

        // stop if the maximum difference of gradients is less than grad_tol
        double grad_tol_ = 0.0001;

        // maximum number of iterations
        size_t max_t_ = std::numeric_limits<size_t>::max();
    };

    TwoClassSVM(RandEngine const & randengine = RandEngine::global())
        : randengine_(randengine)
    {}

    /// \brief Train the SVM.
    /// \param features: the features
    /// \param labels: the labels
    /// \param U: upper bound for alpha
    /// \param B: value of the bias feature (added after normalization)
    /// \param stop: the stopping criteria
    template <typename FEATURES, typename LABELS>
    void train(
            FEATURES const & features,
            LABELS const & labels,
            double const U = 1.0,
            double const B = 1.5,
            StoppingCriteria const & stop = StoppingCriteria()
    );

    /// \brief Predict with the SVM.
    /// \param features: the features
    /// \param labels[out]: the predicted labels
    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & features,
            LABELS & labels
    ) const;

    /// \brief Transform the labels to +1 and -1.
    /// \param labels_in: the labels that should be transformed
    /// \param labels_out[out]: the transformed labels
    template <typename LABELS>
    void transform_external_labels(
            LABELS const & labels_in,
            MultiArrayView<1, int> & labels_out
    ) const;

    /// \brief Setter for the alpha vector.
    template <typename ARRAY>
    void set_alpha(ARRAY const & new_alpha)
    {
        alpha_.reshape(Shape1(new_alpha.size()));
        size_t i = 0;
        for (auto it = new_alpha.begin(); it != new_alpha.end(); ++it, ++i)
        {
            alpha_(i) = *it;
        }
    }

    /// \brief Getter for the alpha vector.
    MultiArray<1, double> const & alpha() const
    {
        return alpha_;
    }

protected:

    /// \brief Find mean and standard deviation of the given features and save them.
    /// \param features: the features
    template <typename FEATURES>
    void find_normalization(
            FEATURES const & features
    );

    /// \brief Apply the current normalization on the given features and add the bias feature.
    /// \param features_in: the features that shall be normalized
    /// \param features_out[out]: the normalized features
    template <typename FEATURES>
    void apply_normalization(
            FEATURES const & features_in,
            MultiArray<2, double> & features_out
    ) const;

    /// \brief The random engine.
    RandEngine const & randengine_;

    /// \brief The labels that were found in training.
    std::vector<LabelType> distinct_labels_;

    /// \brief The alpha vector that is computed in training.
    MultiArray<1, double> alpha_;

    /// \brief The beta vector that is computed in training.
    MultiArray<1, double> beta_;

    /// \brief The vector with the mean of each feature dimension of the training data.
    MultiArray<1, double> mean_;

    /// \brief The vector with the standard deviation of each feature dimension of the training data.
    MultiArray<1, double> std_dev_;

    /// \brief Value of the bias feature.
    double B_;
};

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::train(
        FEATURES const & features,
        LABELS const & labels,
        double const U,
        double const B,
        StoppingCriteria const & stop
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "TwoClassSVM::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "TwoClassSVM::train(): Wrong label type.");

    size_t const num_instances = features.shape()[0];
    size_t num_features = features.shape()[1]+1; // +1 for the bias feature

    // Save the bias feature.
    B_ = B;

    // Find the distinct labels.
    auto dlabels = std::set<LabelType>(labels.begin(), labels.end());
    vigra_precondition(dlabels.size() == 1 || dlabels.size() == 2,
                       "TwoClassSVM::train(): Number of classes must be 1 or 2.");
    distinct_labels_.resize(dlabels.size());
    std::copy(dlabels.begin(), dlabels.end(), distinct_labels_.begin());

    // Early out: There is only one class.
    if (distinct_labels_.size() == 1)
    {
        beta_.reshape(Shape1(num_features));
        return;
    }

    // Translate the labels to +1 and -1.
    auto label_ids = MultiArray<1, int>(labels.size());
    transform_external_labels(labels, label_ids);

    // Normalize the features.
    auto normalized_features = MultiArray<2, double>(Shape2(num_instances, num_features));
    find_normalization(features);
    apply_normalization(features, normalized_features);

    // Precompute the squared norm of the instances.
    auto x_squ = MultiArray<1, double>(Shape1(num_instances));
    for (size_t i = 0; i < num_instances; ++i)
    {
        double v = 0.;
        for (size_t j = 0; j < num_features; ++j)
        {
            double f = normalized_features(i, j);
            v += f * f;
        }
        x_squ(i) = v;
    }

    // Initialize alphas and betas.
    beta_.reshape(Shape1(num_features), 0.);
    if (alpha_.size() == num_instances)
    {
        // The alphas are initialized, so we must create the according betas.
        for (size_t j = 0; j < num_features; ++j)
        {
            for (size_t i = 0; i < num_instances; ++i)
            {
                beta_(j) += alpha_(i) * label_ids(i) * normalized_features(i, j);
            }
        }
    }
    else
    {
        // The alphas are not initialized, so all alphas and betas are 0.
        alpha_.reshape(Shape1(num_instances), 0.);
    }

    // Do the SVM loop.
    auto indices = std::vector<size_t>(num_instances);
    std::iota(indices.begin(), indices.end(), 0);
    auto rand_int = UniformIntRandomFunctor<RandEngine>(randengine_);
    for (size_t t = 0; t < stop.max_t_;)
    {
        std::random_shuffle(indices.begin(), indices.end(), rand_int);
        size_t diff_count = 0;
        double min_grad = std::numeric_limits<double>::max();
        double max_grad = std::numeric_limits<double>::lowest();
        for (size_t i : indices)
        {
            // Compute the gradient.
            double v = 0.;
            for (size_t j = 0; j < num_features; ++j)
            {
                v += normalized_features(i, j) * beta_(j);
            }
            auto const grad = label_ids(i) * v - 1;

            // Update alpha
            auto old_alpha = alpha_(i);
            alpha_(i) = std::max(0., std::min(U, alpha_(i) - grad/x_squ(i)));

            // Update beta.
            for (size_t j = 0; j < num_features; ++j)
            {
                beta_(j) += label_ids(i) * normalized_features(i, j) * (alpha_(i) - old_alpha);
            }

            // Compute the projected gradient (for the stopping criteria).
            auto proj_grad = grad;
            if (alpha_(i) <= 0)
                proj_grad = std::min(grad, 0.);
            else if (alpha_(i) >= U)
                proj_grad = std::max(grad, 0.);
            min_grad = std::min(min_grad, proj_grad);
            max_grad = std::max(max_grad, proj_grad);

            // Update the stopping criteria.
            if (std::abs(alpha_(i) - old_alpha) > stop.alpha_tol_)
            {
                ++diff_count;
            }
            ++t;
            if (t >= stop.max_t_)
            {
                break;
            }
        }

        if (max_grad - min_grad < stop.grad_tol_ ||
                diff_count <= stop.max_total_diffs_ ||
                diff_count <= stop.max_relative_diffs_ * num_instances)
        {
            break;
        }
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::predict(
        FEATURES const & features,
        LABELS & labels
) const {
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "TwoClassSVM::predict(): Wrong feature type.");
    static_assert(std::is_convertible<LabelType, typename LABELS::value_type>(),
                  "TwoClassSVM::predict(): Wrong label type.");
    vigra_precondition(distinct_labels_.size() == 1 || distinct_labels_.size() == 2,
                       "TwoClassSVM::predict(): Number of labels found in training must be 1 or 2.");
    vigra_precondition(features.shape()[1]+1 == beta_.size(),
                       "TwoClassSVM::predict(): Wrong number of features.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1]+1;

    // If only one class was found in training, always predict this class.
    if (distinct_labels_.size() == 1)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            labels(i) = distinct_labels_[0];
        }
        return;
    }

    // If two classes were found in training, we must do the "real" SVM prediction.
    auto normalized_features = MultiArray<2, double>(Shape2(num_instances, num_features));
    apply_normalization(features, normalized_features);
    for (size_t i = 0; i < num_instances; ++i)
    {
        double v = 0;
        for (size_t j = 0; j < num_features; ++j)
        {
            v += normalized_features(i, j) * beta_(j);
        }
        size_t index = (v >= 0) ? 0 : 1; // if v >= 0 then we use label +1, which has index 0 in distinct_labels_, else we use the label with index 1
        labels(i) = distinct_labels_[index];
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename LABELS>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::transform_external_labels(
        LABELS const & labels_in,
        MultiArrayView<1, int> & labels_out
) const {
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "TwoClassSVM::transform_external_labels(): Wrong label type.");
    vigra_precondition(labels_in.size() == labels_out.size(),
                       "TwoClassSVM::transform_external_labels(): Shape mismatch.");
    vigra_precondition(distinct_labels_.size() == 2,
                       "TwoClassSVM::transform_external_labels(): Number of labels found in training must be 2.");

    std::map<FeatureType, int> label_ids {
        {distinct_labels_[0], 1},
        {distinct_labels_[1], -1}
    };
    for (size_t i = 0; i < labels_in.size(); ++i)
    {
        labels_out(i) = label_ids[labels_in(i)];
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::find_normalization(
        FEATURES const & features
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "TwoClassSVM::find_normalization(): Wrong feature type.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];

    // Find the mean.
    mean_.reshape(Shape1(num_features), 0.);
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            mean_[j] += features(i, j);
        }
        mean_[j] /= num_instances;
    }

    // Find the standard deviation.
    std_dev_.reshape(Shape1(num_features), 0.);
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            double v = features(i, j) - mean_[j];
            std_dev_[j] += v*v;
        }
        std_dev_[j] = std::sqrt(std_dev_[j] / (num_instances-1.));
        if (std_dev_[j] == 0)
        {
            std_dev_[j] = 1e-6;
        }
    }
}

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::apply_normalization(
        FEATURES const & features_in,
        MultiArray<2, double> & features_out
) const {
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "TwoClassSVM::normalize(): Wrong feature type.");

    size_t const num_instances = features_in.shape()[0];
    size_t const num_features = features_in.shape()[1];

    // Normalize the data.
    features_out.reshape(Shape2(num_instances, num_features+1));
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            features_out(i, j) = (features_in(i, j) - mean_[j]) / std_dev_[j];
        }
    }

    // Add the bias feature.
    for (size_t i = 0; i < num_instances; ++i)
    {
        features_out(i, num_features) = B_;
    }
}



/// \brief Divide and Conquer SVM [Hsieh et al. 2014].
template <typename SVM>
class ClusteredTwoClassSVM
{
public:

    typedef typename SVM::FeatureType FeatureType;
    typedef typename SVM::LabelType LabelType;
    typedef typename SVM::RandEngine RandEngine;
    typedef typename SVM::StoppingCriteria StoppingCriteria;

    ClusteredTwoClassSVM(
            size_t const rounds,
            size_t const k,
            size_t const num_clustering_samples,
            RandEngine const & randengine = RandEngine::global()
    )   : rounds_(rounds),
          k_(k),
          num_clustering_samples_(num_clustering_samples),
          randengine_(randengine)
    {}

    /// \brief Train the SVM.
    /// \param features: the features
    /// \param labels: the labels
    /// \param U: upper bound for alpha
    /// \param B: value of the bias feature (added after normalization)
    /// \param stop: the stopping criteria
    template <typename FEATURES, typename LABELS>
    void train(
            FEATURES const & features,
            LABELS const & labels,
            double const U = 1.0,
            double const B = 1.5,
            StoppingCriteria const & stop = StoppingCriteria()
    );

    /// \brief Predict with the SVM.
    /// \param features: the features
    /// \param labels[out]: the predicted labels
    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & features,
            LABELS & labels
    ) const;

    void set_alpha(MultiArrayView<1, double> const & new_alpha)
    {
        alpha_ = new_alpha;
    }

    MultiArray<1, double> const & alpha() const
    {
        return alpha_;
    }

protected:

    void create_cluster_sample(
            size_t const round,
            size_t const num_instances,
            std::vector<size_t> & sample
    ) const;

    /// \brief Number of rounds.
    size_t const rounds_;

    /// \brief Use k^(rounds-1-i) clusters in round i.
    size_t const k_;

    /// \brief Number of instances that are used to compute the clusters.
    size_t const num_clustering_samples_;

    /// \brief The random engine.
    RandEngine const & randengine_;

    /// \brief The alpha vector that is computed in training.
    MultiArray<1, double> alpha_;

    /// \brief The final SVM that is used for the prediction.
    SVM final_svm_;
};

template <typename SVM>
template <typename FEATURES, typename LABELS>
void ClusteredTwoClassSVM<SVM>::train(
        FEATURES const & features,
        LABELS const & labels,
        double const U,
        double const B,
        StoppingCriteria const & stop
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "ClusteredTwoClassSVM::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "ClusteredTwoClassSVM::train(): Wrong label type.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];

    // Initialize the alphas.
    if (alpha_.size() != num_instances)
    {
        alpha_.reshape(Shape1(num_instances), 0.);
    }

    for (size_t l = 0; l+1 < rounds_; ++l)
    {
        size_t const num_clusters = std::pow(k_, rounds_ - 1 - l);

        // Draw a random sample of the instances for the clustering.
        std::vector<size_t> sample_indices;
        create_cluster_sample(l, num_instances, sample_indices);

        // Do the clustering.
        std::vector<size_t> instance_clusters;
        KMeansStoppingCriteria kmeans_stop;
        kmeans(features, num_clusters, instance_clusters, kmeans_stop, sample_indices);
        vigra_assert(instance_clusters.size() == num_instances,
                     "ClusteredTwoClassSVM::train(): The kmeans algorithm produced the wrong number of instances.");

        // Create the instance views for the sub SVMs.
        std::vector<std::vector<size_t> > sub_instance_indices(num_clusters);
        for (size_t i = 0; i < num_instances; ++i)
        {
            sub_instance_indices[instance_clusters[i]].push_back(i);
        }

        for (size_t c = 0; c < num_clusters; ++c)
        {
            // Get the sub problem.
            auto const & indices = sub_instance_indices[c];
            detail::LineView<FEATURES> sub_features(features, indices);
            detail::LineView<LABELS> sub_labels(labels, indices);
            detail::LineView<MultiArray<1, double> > sub_alpha(alpha_, indices);

            // Train the SVM on the sub problem.
            SVM svm;
            svm.set_alpha(sub_alpha);
            svm.train(sub_features, sub_labels, U, B, stop);

            // Update the alphas.
            auto const & svm_alpha = svm.alpha();
            vigra_assert(svm_alpha.size() == indices.size(),
                         "ClusteredTwoClassSVM::train(): Sub SVM has wrong number of alphas.");
            for (size_t i = 0; i < svm_alpha.size(); ++i)
            {
                alpha_(indices[i]) = svm_alpha(i);
            }
        }
    }

    // Refine the solution by running an SVM on all instances with alpha > 0.
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < num_instances; ++i)
        {
            if (alpha_(i) > 0)
            {
                indices.push_back(i);
            }
        }
        detail::LineView<FEATURES> sub_features(features, indices);
        detail::LineView<LABELS> sub_labels(labels, indices);
        detail::LineView<MultiArray<1, double> > sub_alpha(alpha_, indices);
        SVM svm;
        svm.set_alpha(sub_alpha);
        svm.train(sub_features, sub_labels, U, B, stop);
        auto const & svm_alpha = svm.alpha();
        vigra_assert(svm_alpha.size() == indices.size(),
                     "ClusteredTwoClassSVM::train(): Sub SVM has wrong number of alphas.");
        for (size_t i = 0; i < svm_alpha.size(); ++i)
        {
            alpha_(indices[i]) = svm_alpha(i);
        }
    }

    // Run the final SVM on all instances.
    {
        final_svm_.set_alpha(alpha_);
        final_svm_.train(features, labels, U, B, stop);
        auto const & svm_alpha = final_svm_.alpha();
        vigra_assert(svm_alpha.size() == num_instances,
                     "ClusteredTwoClassSVM::train(): Final SVM has wrong number of alphas.");
        for (size_t i = 0; i < num_instances; ++i)
        {
            alpha_(i) = svm_alpha(i);
        }
    }
}

template <typename SVM>
template <typename FEATURES, typename LABELS>
void ClusteredTwoClassSVM<SVM>::predict(
        FEATURES const & features,
        LABELS & labels
) const {
    final_svm_.predict(features, labels);
}

template <typename SVM>
void ClusteredTwoClassSVM<SVM>::create_cluster_sample(
        size_t const round,
        size_t const num_instances,
        std::vector<size_t> & sample
) const {
    if (round == 0)
    {
        // First round: Consider all instances.
        sample.resize(num_instances);
        std::iota(sample.begin(), sample.end(), 0);
    }
    else
    {
        // Other rounds: Consider all instances with alpha > 0.
        for (size_t i = 0; i < num_instances; ++i)
        {
            if (alpha_(i) > 0)
            {
                sample.push_back(i);
            }
        }
    }

    // Do the sampling.
    if (sample.size() > num_clustering_samples_)
    {
        UniformIntRandomFunctor<MersenneTwister> rand(randengine_);
        for (size_t i = 0; i < num_clustering_samples_; ++i)
        {
            size_t ii = i+rand(sample.size()-i);
            std::swap(sample[i], sample[ii]);
        }
        sample.resize(num_clustering_samples_);
    }
}



} // namespace vigra



#endif
