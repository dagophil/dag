#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>

#include <vigra/randomforest.hxx>



void test_randomforest0()
{
    using namespace vigra;

    typedef Forest1<DAGraph0> Forest;
    typedef FeatureGetter<float> Features;
    typedef LabelGetter<UInt8> Labels;

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
        MultiArray<2, float> train_x;
        MultiArray<1, UInt8> train_y;
        MultiArray<2, float> test_x;
        MultiArray<1, UInt8> test_y;
        HDF5ImportInfo info(train_filename.c_str(), "data");
        train_x.reshape(Shape2(info.shape().begin()));
        readHDF5(info, train_x);
        info = HDF5ImportInfo(train_filename.c_str(), "labels");
        train_y.reshape(Shape1(info.shape().begin()));
        readHDF5(info, train_y);
        info = HDF5ImportInfo(test_filename.c_str(), "data");
        test_x.reshape(Shape2(info.shape().begin()));
        readHDF5(info, test_x);
        info = HDF5ImportInfo(test_filename.c_str(), "labels");
        test_y.reshape(Shape1(info.shape().begin()));
        readHDF5(info, test_y);

        // Train a random forest.
        RandomForest0<Forest, Features, Labels> rf;
        Features train_feats(train_x);
        Labels train_labels(train_y);
        rf.train(train_feats, train_labels, 1);
    }

    std::cout << "test_randomforest0(): Success!" << std::endl;
}

int main()
{
    test_randomforest0();
}
