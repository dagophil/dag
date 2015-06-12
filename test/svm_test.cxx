#include <iostream>

#include <vigra/svm.hxx>
#include <vigra/feature_getter.hxx>
#include "data_utility.hxx"

void test_svm()
{
    using namespace vigra;

    std::cout << "called test_svm()" << std::endl;

    {
        std::cout << "Running SVM on toy data" << std::endl;

        typedef double FeatureType;
        typedef UInt8 LabelType;

        std::vector<FeatureType> train_data_x {
            9,  9, 10.5,  12, 12,
            2, -1,  0.5, 2, -1
        };
        std::vector<LabelType> train_data_y {
            0, 0, 0, 1, 1
        };
        std::vector<FeatureType> test_data_x {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0
        };
        std::vector<LabelType> test_data_y {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
        };

        // Transform the data into multi arrays.
        MultiArray<2, FeatureType> train_x(Shape2(5, 2));
        for (size_t i = 0; i < train_data_x.size(); ++i)
            train_x[i] = train_data_x[i];

        MultiArray<1, LabelType> train_y(train_data_y.size());
        for (size_t i = 0; i < train_data_y.size(); ++i)
            train_y[i] = train_data_y[i];

        MultiArray<2, FeatureType> test_x(Shape2(15, 2));
        for (size_t i = 0; i < test_data_x.size(); ++i)
            test_x[i] = test_data_x[i];

        MultiArray<1, LabelType> test_y(test_data_y.size());
        for (size_t i = 0; i < test_data_y.size(); ++i)
            test_y[i] = test_data_y[i];

        // Train the SVM.
        TwoClassSVM<FeatureType, LabelType> svm;
        svm.train(train_x, train_y);

        // Predict with the SVM.
        MultiArray<1, LabelType> pred_y(15);
        svm.predict(test_x, pred_y);

        // Count the correct predicted instances.
        size_t count = 0;
        for (size_t i = 0; i < test_y.size(); ++i)
        {
            if (pred_y(i) == test_y(i))
                ++count;
        }
        std::cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
    }

    {
        std::cout << "Running SVM on MNIST 5 vs 8" << std::endl;

        typedef double FeatureType;
        typedef UInt8 LabelType;
        typedef TwoClassSVM<FeatureType, LabelType> SVM;

        // Load the data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<LabelType> labels = {5, 8};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        // Train a SVM.
        SVM::Options opt;
        opt.max_relative_diffs_ = 0.02;
//        opt.grad_tol_ = 0.001;
        SVM svm(opt);
        svm.train(train_x, train_y);

        // Predict with the SVM.
        MultiArray<1, LabelType> pred_y(test_y.shape());
        svm.predict(test_x, pred_y);

        // Count the correct predicted instances.
        size_t count = 0;
        for (size_t i = 0; i < test_y.size(); ++i)
        {
            if (pred_y(i) == test_y(i))
                ++count;
        }
        std::cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
    }

    std::cout << "finished test_svm()" << std::endl;
}

void test_sparse_svm()
{
    using namespace vigra;

    {
        std::cout << "Running sparse SVM on MNIST 5 vs 8" << std::endl;

        typedef double FeatureType;
        typedef UInt8 LabelType;
        typedef TwoClassSVM<FeatureType, LabelType> SVM;

        // Load the data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<LabelType> labels = {5, 8};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);
        SparseFeatureGetter<FeatureType> train_features(train_x);
        SparseFeatureGetter<FeatureType> test_features(test_x);

        // Train a SVM.
        SVM::Options opt;
        opt.max_relative_diffs_ = 0.02;
//        opt.grad_tol_ = 0.001;
        SVM svm(opt);
        svm.train(train_features, train_y);

        // Predict with the SVM.
        MultiArray<1, LabelType> pred_y(test_y.shape());
        svm.predict(test_features, pred_y);

        // Count the correct predicted instances.
        size_t count = 0;
        for (size_t i = 0; i < test_y.size(); ++i)
        {
            if (pred_y(i) == test_y(i))
                ++count;
        }
        std::cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
    }

    std::cout << "test_sparse_svm(): Success!" << std::endl;
}

void test_clustered_svm()
{
    using namespace vigra;

    {
        std::cout << "Running clustered SVM on MNIST 5 vs 8" << std::endl;

        typedef double FeatureType;
        typedef UInt8 LabelType;
        typedef TwoClassSVM<FeatureType, LabelType> RegularSVM;
        typedef ClusteredTwoClassSVM<RegularSVM> SVM;

        // Load the data.
        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
        std::vector<LabelType> labels = {5, 8};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        // Train a SVM.
        SVM::Options opt;
//        opt.max_relative_diffs_ = 0.02;
        opt.grad_tol_ = 0.001;
        SVM svm(5, 4, 2000, opt);
        svm.train(train_x, train_y);

        // Predict with the SVM.
        MultiArray<1, LabelType> pred_y(test_y.shape());
        svm.predict(test_x, pred_y);

        // Count the correct predicted instances.
        size_t count = 0;
        for (size_t i = 0; i < test_y.size(); ++i)
        {
            if (pred_y(i) == test_y(i))
                ++count;
        }
        std::cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << std::endl;
    }

    std::cout << "finished test_clustered_svm()" << std::endl;
}


int main()
{
    test_svm();
    test_sparse_svm();
//    test_clustered_svm();
}
