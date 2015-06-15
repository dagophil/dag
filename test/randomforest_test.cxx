#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <unordered_set>
#include <chrono>

#include <vigra/randomforest.hxx>
#include "data_utility.hxx"



// TIC TOC macros to measure time.
std::chrono::steady_clock::time_point start, end;
double sec;
#define TIC start = std::chrono::steady_clock::now();
#define TOC(msg) end = std::chrono::steady_clock::now(); sec = std::chrono::duration<double>(end-start).count(); std::cout << msg << ": " << sec << " seconds" << std::endl;



void test_randomforest0()
{
    using namespace vigra;

    typedef float FeatureType;
    typedef UInt8 LabelType;
    typedef FeatureGetter<FeatureType> Features;
    typedef LabelGetter<LabelType> Labels;
    typedef BootstrapSampler Sampler;
    typedef PurityTermination Termination;
    typedef RandomSplit<GiniScorer> SplitFunctor;

    // Test sample_with_replacement().
    {
        // Draw k samples from the range [0, ..., n-1].
        size_t n = 30;
        size_t k = 10;

        std::vector<int> v_in(n);
        std::iota(v_in.begin(), v_in.end(), 0);

        std::vector<int> v_sample;
        detail::sample_with_replacement(k, v_in.begin(), v_in.end(), std::back_inserter(v_sample));
        vigra_assert(v_sample.size() == k, "sample_with_replacement(): Output has wrong size.");

        // TODO: Check that the output is random and values may occur multiple times.
    }

    // Test sample_without_replacement().
    {
        // Draw k samples from the range [0, ..., n-1].
        size_t n = 30;
        size_t k = 20;

        std::vector<int> v_in(n);
        std::iota(v_in.begin(), v_in.end(), 0);

        std::vector<int> v_sample;
        detail::sample_without_replacement(k, v_in.begin(), v_in.end(), std::back_inserter(v_sample));
        vigra_assert(v_sample.size() == k, "sample_without_replacement(): Output has wrong size.");
        std::unordered_set<int> s(v_sample.begin(), v_sample.end());
        vigra_assert(s.size() == k, "sample_without_replacement(): Elements may only occur once.");

        // TODO: Check that the output is random.
    }

    // Build a random forest.
    {
        // Load some data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<LabelType> labels = {3, 8};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        /*
        // Old implementation.
        vigra::RandomForestOptions RFOPTIONS;
        RFOPTIONS.tree_count(100);
        vigra::RandomForest<> RF(RFOPTIONS);
        MultiArrayView<2, UInt8> train_y_subb(Shape2(train_y_sub.size(), 1), train_y_sub.data());
        RF.learn(train_x_sub, train_y_subb);
        MultiArray<2, UInt8> pred_yy(Shape2(test_y_sub.size(), 1));
        RF.predictLabels(test_x_sub, pred_yy);
        size_t countt = 0;
        for (size_t i = 0; i < test_y_sub.size(); ++i)
        {
            if (pred_yy[i] == test_y_sub[i])
                ++countt;
        }
        std::cout << "Performance: " << (countt / ((float) pred_yy.size())) << " (" << countt << " of " << pred_yy.size() << ")" << std::endl;
        */

        // Train a random forest.
//         MersenneTwister randengine(0);
//         RandomForest0<FeatureType, LabelType> rf(randengine);
        RandomForest0<FeatureType, LabelType> rf;
        Features train_feats(train_x);
        Labels train_labels(train_y);

        TIC;
        rf.train<Features, Labels, Sampler, Termination, SplitFunctor>(
                    train_feats, train_labels, 100
        );
        TOC("Random forest training");

        // Predict using the forest.
        MultiArray<1, LabelType> pred_y(test_y.shape());
        Features test_feats(test_x);
        TIC;
        rf.predict(test_feats, pred_y);
        TOC("Random forest prediction");

        // Count the correct predicted instances.
        size_t count = 0;
        for (size_t i = 0; i < test_y.size(); ++i)
        {
            if (pred_y[i] == test_y[i])
                ++count;
        }
        std::cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
    }

    std::cout << "test_randomforest0(): Success!" << std::endl;
}

void test_globallyrefinedrf()
{
    using namespace vigra;

    typedef float FeatureType;
    typedef UInt8 LabelType;
    typedef FeatureGetter<FeatureType> Features;
    typedef LabelGetter<LabelType> Labels;
    typedef BootstrapSampler Sampler;
    typedef PurityTermination Termination;
    typedef RandomSplit<GiniScorer> SplitFunctor;
    typedef RandomForest0<FeatureType, LabelType> RandomForest;

    {
        // Load some data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<LabelType> labels = {3, 8};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        // Train a random forest.
        RandomForest rf;
        TIC;
        rf.train<MultiArray<2, FeatureType>, MultiArray<1, LabelType>, Sampler, Termination, SplitFunctor>(
                    train_x, train_y, 100
        );
        TOC("Random forest training")

        // Predict on the test set (for comparison).
        {
            MultiArray<1, LabelType> pred_y(test_y.shape());
            Features test_feats(test_x);
            TIC;
            rf.predict(test_feats, pred_y);
            TOC("Random forest prediction");

            // Count the correct predicted instances.
            size_t count = 0;
            for (size_t i = 0; i < test_y.size(); ++i)
            {
                if (pred_y[i] == test_y[i])
                    ++count;
            }
            std::cout << "RF performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
        }

        // Train a globally refined random forest.
        GloballyRefinedRandomForest<RandomForest> grrf(rf);
        TIC;
        grrf.train(train_x, train_y);
        TOC("Global refinement");

        // Predict on the test set.
        {
            // Predict using the forest.
            MultiArray<1, LabelType> pred_y(test_y.shape());
            TIC;
            grrf.predict(test_x, pred_y);
            TOC("Refined random forest prediction");

            // Count the correct predicted instances.
            size_t count = 0;
            for (size_t i = 0; i < test_y.size(); ++i)
            {
                if (pred_y[i] == test_y[i])
                    ++count;
            }
            std::cout << "GRRF performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
        }
    }
}



int main()
{
//    test_randomforest0();
    test_globallyrefinedrf();
}
