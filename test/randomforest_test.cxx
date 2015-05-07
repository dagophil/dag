#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <unordered_set>

#include <vigra/randomforest.hxx>



template <typename S, typename T>
void load_data(
        std::string const & train_filename,
        std::string const & test_filename,
        vigra::MultiArray<2, S> & train_x,
        vigra::MultiArray<1, T> & train_y,
        vigra::MultiArray<2, S> & test_x,
        vigra::MultiArray<1, T> & test_y,
        std::vector<T> const & labels = {}
){
    using namespace vigra;

    // Load the data.
    MultiArray<2, S> tmp_train_x;
    MultiArray<1, T> tmp_train_y;
    MultiArray<2, S> tmp_test_x;
    MultiArray<1, T> tmp_test_y;
    HDF5ImportInfo info(train_filename.c_str(), "data");
    tmp_train_x.reshape(Shape2(info.shape().begin()));
    readHDF5(info, tmp_train_x);
    info = HDF5ImportInfo(train_filename.c_str(), "labels");
    tmp_train_y.reshape(Shape1(info.shape().begin()));
    readHDF5(info, tmp_train_y);
    info = HDF5ImportInfo(test_filename.c_str(), "data");
    tmp_test_x.reshape(Shape2(info.shape().begin()));
    readHDF5(info, tmp_test_x);
    info = HDF5ImportInfo(test_filename.c_str(), "labels");
    tmp_test_y.reshape(Shape1(info.shape().begin()));
    readHDF5(info, tmp_test_y);

    vigra_assert(tmp_train_x.shape()[0] == tmp_train_y.size(), "Wrong number of training instances.");
    vigra_assert(tmp_test_x.shape()[0] == tmp_test_y.size(), "Wrong number of test instances.");

    if (labels.size() == 0)
    {
        train_x = tmp_train_x;
        train_y = tmp_train_y;
        test_x = tmp_test_x;
        test_y = tmp_test_y;
        return;
    }

    // Restrict the training data to the given label subset.
    std::vector<size_t> train_indices;
    for (size_t i = 0; i < tmp_train_y.size(); ++i)
    {
        for (auto const & label : labels)
        {
            if (tmp_train_y[i] == label)
            {
                train_indices.push_back(i);
                break;
            }
        }
    }
    train_x.reshape(Shape2(train_indices.size(), tmp_train_x.shape()[1]));
    train_y.reshape(Shape1(train_indices.size()));
    for (size_t i = 0; i < train_x.shape()[0]; ++i)
    {
        for (size_t k = 0; k < train_x.shape()[1]; ++k)
        {
            train_x(i, k) = tmp_train_x(train_indices[i], k);
        }
        train_y[i] = tmp_train_y[train_indices[i]];
    }

    // Restrict the test data to the given label subset.
    std::vector<size_t> test_indices;
    for (size_t i = 0; i < tmp_test_y.size(); ++i)
    {
        for (auto const & label : labels)
        {
            if (tmp_test_y[i] == label)
            {
                test_indices.push_back(i);
                break;
            }
        }
    }
    test_x.reshape(Shape2(test_indices.size(), tmp_test_x.shape()[1]));
    test_y.reshape(Shape1(test_indices.size()));
    for (size_t i = 0; i < test_x.shape()[0]; ++i)
    {
        for (size_t k = 0; k < test_x.shape()[1]; ++k)
        {
            test_x(i, k) = tmp_test_x(test_indices[i], k);
        }
        test_y[i] = tmp_test_y[test_indices[i]];
    }
}


void test_randomforest0()
{
    using namespace vigra;

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

    // Test the split iterator.
    {
        typedef detail::SplitIterator<std::vector<float>::iterator> FloatSplitIter;
        std::vector<float> v {1.f, 3.f, 3.f, 4.f, 4.f, 4.f, 8.f};
        std::vector<float> splits_expected {2.f, 3.5f, 6.f};
        std::vector<float> splits;
        for (FloatSplitIter it(v.begin(), v.end()); it != v.end(); ++it)
            splits.push_back(*it);
        vigra_assert(splits.size() == splits_expected.size(), "SplitIterator: Wrong number of splits.");
        for (size_t i = 0; i < splits.size(); ++i)
            vigra_assert(splits[i] == splits_expected[i], "SplitIterator: The splits are wrong.");

        // Test border cases: Vector of size 0 and vecto of size 1.
        v.clear();
        splits_expected.clear();
        splits.clear();
        for (FloatSplitIter it(v.begin(), v.end()); it != v.end(); ++it)
            splits.push_back(*it);
        vigra_assert(splits.size() == 0, "SplitIterator: Error on empty vector.");
        v.push_back(1.f);
        for (FloatSplitIter it(v.begin(), v.end()); it != v.end(); ++it)
            splits.push_back(*it);
        vigra_assert(splits.size() == 0, "SplitIterator: Error on vector with size one.");
    }

    // Build a random forest.
    {
        // Load some data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<UInt8> labels = {3, 8};
        MultiArray<2, float> train_x;
        MultiArray<1, UInt8> train_y;
        MultiArray<2, float> test_x;
        MultiArray<1, UInt8> test_y;
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
        RandomForest0<Features, Labels> rf;
        Features train_feats(train_x);
        Labels train_labels(train_y);
        rf.train(train_feats, train_labels, 100);

        // Predict using the forest.
        MultiArray<1, UInt8> pred_y(test_y.shape());
        Features test_feats(test_x);
        rf.predict(test_feats, pred_y);

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

void test_modularrandomforest()
{
    using namespace vigra;

    typedef float S;
    typedef UInt8 T;
    typedef FeatureGetter<S> Features;
    typedef LabelGetter<T> Labels;

    {
        // Load some data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<T> labels = {3, 8};
        MultiArray<2, S> train_x;
        MultiArray<1, T> train_y;
        MultiArray<2, S> test_x;
        MultiArray<1, T> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        ModularRandomForest<S, T> rf;
        Features train_feats(train_x);
        Labels train_labels(train_y);
        rf.train(train_feats, train_labels, 100);

        Features test_feats(test_x);
        MultiArray<1, T> pred_y;
        rf.predict(test_feats, pred_y);
    }





    std::cout << "test_modularrandomforest(): Success!" << std::endl;
}




int main()
{
//    test_randomforest0();
    test_modularrandomforest();
}
