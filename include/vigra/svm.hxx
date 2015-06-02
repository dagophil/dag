#ifndef VIGRA_SVM_HXX
#define VIGRA_SVM_HXX

#include <vector>
#include <set>
#include <map>

#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>



namespace vigra
{



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

    RandEngine const & randengine_;
    std::vector<LabelType> distinct_labels_;
    MultiArray<1, double> beta_;
    MultiArray<1, double> mean_;
    MultiArray<1, double> std_dev_;
};



template <typename FEATURES>
class DotProductEvaluater
{
public:
    typedef FEATURES Features;
    DotProductEvaluater(Features const & features)
        : features_(features)
    {}
    double operator()(size_t i, size_t j) const
    {
        double v = 0;
        for (size_t k = 0; k < features_.shape()[1]; ++k)
        {
            v += features_(i, k) * features_(j, k);
        }
    }
protected:
    Features const & features_;
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

    double const U = 1.0; // TODO: Make this a parameter.
    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];
    double const tol = 0.0001;

    // Find the distinct labels.
    auto dlabels = std::set<LabelType>(labels.begin(), labels.end());
    vigra_precondition(dlabels.size() == 2, "TwoClassSVM::train(): Number of classes must be 2.");
    distinct_labels_.resize(dlabels.size());
    std::copy(dlabels.begin(), dlabels.end(), distinct_labels_.begin());

    // Translate the labels to +1 and -1.
    auto label_ids = MultiArray<1, int>(labels.size());
    transform_external_labels(labels, label_ids);

    // Normalize the data.
    mean_.reshape(Shape1(num_features), 0.);
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            mean_[j] += features(i, j);
        }
        mean_[j] /= num_instances;
    }
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
            std::cout << "Warning: Standard deviation is zero, division by zero will occur." << std::endl;
        }
    }
    auto normalized_features = MultiArray<2, double>(Shape2(num_instances, num_features));
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            normalized_features(i, j) = (features(i, j) - mean_[j]) / std_dev_[j];
        }
    }

    // Initialize the alphas and betas with 0.
    auto alpha = MultiArray<1, double>(Shape1(num_instances), 0.);
    beta_.reshape(Shape1(num_features), 0.);

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

    // Do the SVM loop.
    auto indices = std::vector<size_t>(num_instances);
    std::iota(indices.begin(), indices.end(), 0);
    auto rand_int = UniformIntRandomFunctor<RandEngine>(randengine_);
    for (size_t t = 0; t < max_t;)
    {
        std::random_shuffle(indices.begin(), indices.end(), rand_int);
        size_t diff_count = 0;
        for (size_t i : indices)
        {
            // Compute the scalar products v = x_i * beta.
            double v = 0.;
            for (size_t j = 0; j < num_features; ++j)
            {
                v += normalized_features(i, j) * beta_(j);
            }

            // Update alpha.
            auto gamma = label_ids(i) * v - 1;
            auto old_alpha = alpha(i);
            alpha(i) = std::max(0., std::min(U, alpha(i) - gamma/x_squ(i)));

            // Update beta.
            if (std::abs(alpha(i) - old_alpha) > tol)
            {
                ++diff_count;
                for (size_t j = 0; j < num_features; ++j)
                {
                    beta_(j) += label_ids(j) * normalized_features(i, j) * (alpha(i) - old_alpha);
                }
            }



            ++t;
            if (t % 500000 == 0)
            {
                std::cout << t << std::endl;
            }
            if (t >= max_t)
            {
                break;
            }
        }

        std::cout << "diffs: " << diff_count << std::endl;
        if (1000*diff_count < num_instances)
        {
            break;
        }
    }

    size_t c0 = 0;
    size_t c1 = 0;
    for (size_t i = 0; i < alpha.size(); ++i)
    {
        if (alpha(i) == 0)
            ++c0;
        if (alpha(i) == U)
            ++c1;
    }
    std::cout << c0 << " of " << alpha.size() << " values == 0" << std::endl;
    std::cout << c1 << " of " << alpha.size() << " values == C" << std::endl;
    std::cout << (c0+c1) << " of " << alpha.size() << " values are 0 or C" << std::endl;

//    std::cout << "alpha: ";
//    for (size_t i = 0; i < alpha.size(); ++i) std::cout << alpha(i) << ", ";
//    std::cout << std::endl;
//    std::cout << "beta: ";
//    for (size_t i = 0; i < beta_.size(); ++i) std::cout << beta_(i) << ", ";
//    std::cout << std::endl;
//    std::cout << "classes: " << (int)distinct_labels_[0] << ": +1, " << (int)distinct_labels_[1] << ": -1" << std::endl;
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
    vigra_precondition(features.shape()[1] == beta_.size(),
                       "TwoClassSVM::predict(): Wrong number of features.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];

    for (size_t i = 0; i < num_instances; ++i)
    {
        double v = 0;
        for (size_t j = 0; j < num_features; ++j)
        {
            v += (features(i, j) - mean_[j]) / std_dev_[j] * beta_(j);
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



} // namespace vigra



#endif
