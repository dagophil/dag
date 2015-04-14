#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>

#include <vigra/randomforest.hxx>



void test_randomforest0()
{
    using namespace vigra;

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

    RandomForestBase0<Forest1<DAGraph0> > rf;

    std::cout << "test_randomforest0(): Success!" << std::endl;
}

int main()
{
    test_randomforest0();
}
