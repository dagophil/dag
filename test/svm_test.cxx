#include <iostream>

#include <vigra/svm.hxx>
#include "data_utility.hxx"

void test_svm()
{
    using namespace vigra;

    std::cout << "called test_svm()" << std::endl;

    {
        typedef double FeatureType;
        typedef UInt8 LabelType;

        // Load the data.
//        std::string train_filename = "/home/philip/data/ml-koethe/train.h5";
//        std::string test_filename = "/home/philip/data/ml-koethe/test.h5";
//        std::vector<LabelType> labels = {3, 8};
        std::string train_filename = "/home/philip/data/liblinear/susy_float64_train.h5";
        std::string test_filename = "/home/philip/data/liblinear/susy_float64_test.h5";
        std::vector<LabelType> labels = {0, 1};
        MultiArray<2, FeatureType> train_x;
        MultiArray<1, LabelType> train_y;
        MultiArray<2, FeatureType> test_x;
        MultiArray<1, LabelType> test_y;
        load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

        // Train a SVM.
        TwoClassSVM<FeatureType, LabelType> svm;
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


int main()
{
    test_svm();
}
