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
            MultiArrayView<1, Int8> & labels_out
    ) const;

protected:

    RandEngine const & randengine_;
    std::vector<LabelType> distinct_labels_;
    MultiArray<1, double> beta_;
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

    // TODO: Add constant feature to simulate beta_0.

    double const C = 1.0; // TODO: Make this a parameter.
    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];

    // Find the distinct labels.
    std::set<LabelType> dlabels(labels.begin(), labels.end());
    vigra_precondition(dlabels.size() == 2, "TwoClassSVM::train(): Number of classes must be 2.");
    distinct_labels_.resize(dlabels.size());
    std::copy(dlabels.begin(), dlabels.end(), distinct_labels_.begin());

    // Translate the labels to +1 and -1.
    MultiArray<1, Int8> label_ids(labels.size());
    transform_external_labels(labels, label_ids);

    // Initialize the alphas with random values.
    // NOTE:
    // The const cast is necessary because the constructor of UniformRandomFunctor
    // does not like a const engine (minor bug in vigra, since afterwards the engine
    // is made const again).
    UniformRandomFunctor<RandEngine> rand(0, C, const_cast<RandEngine &>(randengine_));
    MultiArray<1, double> alpha(num_instances);
    for (size_t i = 0; i < alpha.size(); ++i)
    {
        alpha(i) = rand();
    }

    // Initialize the betas.
    beta_.reshape(Shape1(num_features));
    for (size_t j = 0; j < num_features; ++j)
    {
        for (size_t i = 0; i < num_instances; ++i)
        {
            beta_(j) += alpha(i) * label_ids(i) * features(i, j);
        }
    }

    // Do the SVM loop.
    std::vector<size_t> indices(num_instances);
    std::iota(indices.begin(), indices.end(), 0);
    UniformIntRandomFunctor<RandEngine> rand_int(randengine_);
    for (size_t t = 0; t < max_t;)
    {
        std::random_shuffle(indices.begin(), indices.end(), rand_int);
        bool alpha_changed = false;
        for (size_t i : indices)
        {
            // Compute the scalar products v = x_i * beta and x2 = x_i * x_i.
            double v = 0.;
            double x2 = 0.;
            for (size_t k = 0; k < num_features; ++k)
            {
                v += features(i, k) * beta_(k);
                x2 += features(i, k) * features(i, k);
            }

            // Update alpha.
            auto gamma = label_ids(i) * v - 1;
            auto old_alpha = alpha(i);
            alpha(i) = std::max(0., std::min(C, alpha(i) - gamma/x2));
            if (alpha(i) != old_alpha) // TODO: Maybe use some floating point tolerance here.
            {
                alpha_changed = true;
            }

            // Update beta.
            for (size_t k = 0; k < num_features; ++k)
            {
                beta_(k) += label_ids(k) * features(k) * (alpha(i) - old_alpha);
            }

            ++t;
            if (t >= max_t)
            {
                break;
            }
        }

        if (!alpha_changed)
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
    vigra_precondition(features.shape()[1] == beta_.size(),
                       "TwoClassSVM::predict(): Wrong number of features.");

    size_t const num_instances = features.shape()[0];
    size_t const num_features = features.shape()[1];


    for (size_t i = 0; i < num_instances; ++i)
    {
        double v = 0;
        for (size_t j = 0; j < num_features; ++j)
        {
            v += features(i, j) * beta_(j);
        }
        size_t index = (v >= 0) ? 0 : 1; // index (in distinct_labels_) of label id +1 is 0 and index of label id -1 is 1.
        labels(i) = distinct_labels_[index];
    }
}



template <typename FEATURETYPE, typename LABELTYPE, typename RANDENGINE>
template <typename LABELS>
void TwoClassSVM<FEATURETYPE, LABELTYPE, RANDENGINE>::transform_external_labels(
        LABELS const & labels_in,
        MultiArrayView<1, Int8> & labels_out
) const {
    static_assert(std::is_convertible<typename LABELS::value_type, LabelType>(),
                  "TwoClassSVM::transform_external_labels(): Wrong label type.");
    vigra_precondition(labels_in.size() == labels_out.size(),
                       "TwoClassSVM::transform_external_labels(): Shape mismatch.");
    vigra_precondition(distinct_labels_.size() == 2,
                       "TwoClassSVM::transform_external_labels(): Number of labels found in training must be 2.");

    std::map<FeatureType, Int8> label_ids {
        {distinct_labels_[0], 1},
        {distinct_labels_[1], -1}
    };
    for (size_t i = 0; i < labels_in.size(); ++i)
    {
        labels_out[i] = label_ids[labels_in[i]];
    }
}



} // namespace vigra



#endif
