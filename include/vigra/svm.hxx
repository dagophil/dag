#ifndef VIGRA_SVM_HXX
#define VIGRA_SVM_HXX

#include <vector>
#include <set>
#include <map>

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>



namespace vigra
{



//namespace detail
//{
//    /// \brief Compute the dot product of the given instances.
//    template <typename FEATURES>
//    class DotProductEvaluater
//    {
//    public:
//        typedef FEATURES Features;
//        DotProductEvaluater(Features const & features)
//            : features_(features)
//        {}
//        double operator()(size_t i, size_t j) const
//        {
//            double v = 0;
//            for (size_t k = 0; k < features_.shape()[1]; ++k)
//            {
//                v += features_(i, k) * features_(j, k);
//            }
//        }
//    protected:
//        Features const & features_;
//    };
//}



/// \brief Two class support vector machine using hinge loss and a quadratic regularizer (implemented with "dual coordinate descent" [Hsieh et al. 2008])
template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE = MersenneTwister>
class TwoClassSVM
{
public:

    typedef FEATURETYPE FeatureType;
    typedef LABELTYPE LabelType;
    typedef RANDENGINE RandEngine;

    TwoClassSVM(RandEngine const & randengine = RandEngine::global())
        : randengine_(randengine)
    {}

    template <typename FEATURES, typename LABELS>
    void train(
            FEATURES const & features,
            LABELS const & labels,
            size_t const max_t = std::numeric_limits<size_t>::max()
    );

    template <typename FEATURES, typename LABELS>
    void predict(
            FEATURES const & features,
            LABELS & labels
    ) const;

    /// \brief Transform the labels to +1 and -1.
    template <typename LABELS>
    void transform_external_labels(
            LABELS const & labels_in,
            MultiArrayView<1, int> & labels_out
    ) const;

protected:

    /// \brief Find mean and standard deviation of the given features and save them.
    template <typename FEATURES>
    void find_normalization(
            FEATURES const & features
    );

    /// \brief Apply the current normalization on the given features and add the bias feature.
    template <typename FEATURES>
    void apply_normalization(
            FEATURES const & features_in,
            MultiArray<2, double> & features_out
    ) const;

    RandEngine const & randengine_;
    std::vector<LabelType> distinct_labels_;
    MultiArray<1, double> beta_;
    MultiArray<1, double> mean_;
    MultiArray<1, double> std_dev_;
    double B_; // value of the bias feature
};

template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename FEATURES, typename LABELS>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::train(
        FEATURES const & features,
        LABELS const & labels,
        size_t const max_t
){
    static_assert(std::is_convertible<typename FEATURES::value_type, FeatureType>(),
                  "TwoClassSVM::train(): Wrong feature type.");
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "TwoClassSVM::train(): Wrong label type.");

    size_t const num_instances = features.shape()[0];
    size_t num_features = features.shape()[1];

    double const U = 1.0; // TODO: Make this a parameter.
    B_ = 1.5; // value of the bias feature, TODO: Make this a parameter.

    // stopping criterion: alpha difference
    double const alpha_tol = 0.0001;
    size_t const max_diffs = 0;

    // stopping criterion: gradient difference
    double const grad_tol = 0.0001;

    // Find the distinct labels.
    auto dlabels = std::set<LabelType>(labels.begin(), labels.end());
    vigra_precondition(dlabels.size() == 2, "TwoClassSVM::train(): Number of classes must be 2.");
    distinct_labels_.resize(dlabels.size());
    std::copy(dlabels.begin(), dlabels.end(), distinct_labels_.begin());

    // Translate the labels to +1 and -1.
    auto label_ids = MultiArray<1, int>(labels.size());
    transform_external_labels(labels, label_ids);

    // Normalize the features.
    ++num_features; // increase the number since the bias feature will be added
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

    // Initialize the alphas and betas with 0 and do the SVM loop.
    auto alpha = MultiArray<1, double>(Shape1(num_instances), 0.);
    beta_.reshape(Shape1(num_features), 0.);
    auto indices = std::vector<size_t>(num_instances);
    std::iota(indices.begin(), indices.end(), 0);
    auto rand_int = UniformIntRandomFunctor<RandEngine>(randengine_);
    for (size_t t = 0; t < max_t;)
    {
        std::random_shuffle(indices.begin(), indices.end(), rand_int);
        size_t diff_count = 0;
        double min_grad = std::numeric_limits<double>::max();
        double max_grad = std::numeric_limits<double>::lowest();
        for (size_t i : indices)
        {
            // Compute the scalar products v = x_i * beta.
            double v = 0.;
            for (size_t j = 0; j < num_features; ++j)
            {
                v += normalized_features(i, j) * beta_(j);
            }

            // Find the gradient.
            auto grad = label_ids(i) * v - 1;

            // Find the projected gradient (for the stopping criteria).
            auto proj_grad = grad;
            if (alpha(i) <= 0)
                proj_grad = std::min(grad, 0.);
            else if (alpha(i) >= U)
                proj_grad = std::max(grad, 0.);
            min_grad = std::min(min_grad, proj_grad);
            max_grad = std::max(max_grad, proj_grad);

            // Update alpha
            auto old_alpha = alpha(i);
            alpha(i) = std::max(0., std::min(U, alpha(i) - grad/x_squ(i)));

            // Update beta.
            for (size_t j = 0; j < num_features; ++j)
            {
                beta_(j) += label_ids(i) * normalized_features(i, j) * (alpha(i) - old_alpha);
            }

            if (std::abs(alpha(i) - old_alpha) > alpha_tol)
            {
                ++diff_count;
            }
            ++t;
            if (t >= max_t)
            {
                break;
            }
        }

        // TODO: Remove output.
        std::cout << "min: " << min_grad << ", max: " << max_grad << ", diffs: " << diff_count << std::endl;
        if (max_grad - min_grad < grad_tol)
        {
            break;
        }
        if (diff_count <= max_diffs)
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
    vigra_precondition(distinct_labels_.size() == 2,
                       "TwoClassSVM::predict(): Number of labels found in training must be 2.");
    vigra_precondition(features.shape()[1]+1 == beta_.size(),
                       "TwoClassSVM::predict(): Wrong number of features.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1]+1;

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
            std::cout << "Warning: Standard deviation is zero, so a division by zero will occur." << std::endl;
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



} // namespace vigra



#endif
