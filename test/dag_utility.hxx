#ifndef VIGRA_DAG_UTILITY_HXX
#define VIGRA_DAG_UTILITY_HXX

#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>

void reshapeMyData(){
    using namespace vigra;

    std::string train_filename_in = "/home/philip/data/ml-koethe/old_train.h5";
    std::string test_filename_in = "/home/philip/data/ml-koethe/old_test.h5";
    std::string train_filename_out = "/home/philip/data/ml-koethe/train.h5";
    std::string test_filename_out = "/home/philip/data/ml-koethe/test.h5";

    // Read the training data.
    HDF5ImportInfo info(train_filename_in.c_str(), "images");
    MultiArray<3, float> train_images(Shape3(info.shape().begin()));
    readHDF5(info, train_images);
    MultiArrayView<2, float> train_images_tmp(Shape2(info.shape()[0]*info.shape()[1], info.shape()[2]), train_images.data());
    MultiArrayView<2, float> train_x(train_images_tmp.transpose());
    std::cout << "Read train_x: " << train_x.shape() << std::endl;
    writeHDF5(train_filename_out.c_str(), "data", train_x);

    info = HDF5ImportInfo(train_filename_in.c_str(), "labels");
    MultiArray<1, UInt8> train_y(Shape1(info.shape().begin()));
    readHDF5(info, train_y);
    std::cout << "Read train_y: " << train_y.shape() << std::endl;
    writeHDF5(train_filename_out.c_str(), "labels", train_y);

    // Read the test data.
    info = HDF5ImportInfo(test_filename_in.c_str(), "images");
    MultiArray<3, float> test_images(Shape3(info.shape().begin()));
    readHDF5(info, test_images);
    MultiArrayView<2, float> test_images_tmp(Shape2(info.shape()[0]*info.shape()[1], info.shape()[2]), test_images.data());
    MultiArrayView<2, float> test_x(test_images_tmp.transpose());
    std::cout << "Read test_x: " << test_x.shape() << std::endl;
    writeHDF5(test_filename_out.c_str(), "data", test_x);

    info = HDF5ImportInfo(test_filename_in.c_str(), "labels");
    MultiArray<1, UInt8> test_y(Shape1(info.shape().begin()));
    readHDF5(info, test_y);
    std::cout << "Read test_y: " << test_y.shape() << std::endl;
    writeHDF5(test_filename_out.c_str(), "labels", test_y);
}

#endif
