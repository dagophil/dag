#ifndef DATA_UTILITY_HXX
#define DATA_UTILITY_HXX

#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>



namespace vigra
{



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



} // namespace vigra



#endif
