#include <iostream>
#include <vigra/feature_getter.hxx>

void test_featuregetter()
{
    using namespace vigra;

    // Test operator().
    {
        SparseFeatureGetter<int> features(Shape2(2, 3));
        features(0, 1) = 23;
        features(1, 1) = 15;
        features(1, 2) = 10;

        std::vector<int> expected {
            0, 0, 23, 15, 0, 10
        };

        std::vector<int> feats;
        for (size_t y = 0; y < features.shape()[1]; ++y)
        {
            for (size_t x = 0; x < features.shape()[0]; ++x)
            {
                feats.push_back(features(x, y));
            }
        }
        vigra_assert(feats == expected, "Error in SparseFeatureGetter::operator().");
    }

    // Test MultiArrayView constructor.
    {
        MultiArray<2, int> feature_array(Shape2(2, 3), 0);
        feature_array(0, 1) = 23;
        feature_array(1, 1) = 15;
        feature_array(1, 2) = 10;
        SparseFeatureGetter<int> features(feature_array);

        std::vector<int> expected {
            0, 0, 23, 15, 0, 10
        };

        std::vector<int> feats;
        for (size_t y = 0; y < features.shape()[1]; ++y)
        {
            for (size_t x = 0; x < features.shape()[0]; ++x)
            {
                feats.push_back(features(x, y));
            }
        }
        vigra_assert(feats == expected, "Error in SparseFeatureGetter(MultiArrayView const &).");
    }

    // Test non-zero iterator.
    {
        SparseFeatureGetter<int> features(Shape2(2, 3));
        features(0, 1) = 23;
        features(1, 1) = 15;
        features(1, 2) = 10;

        std::vector<std::tuple<size_t, size_t, int> > expected {
            std::make_tuple(0, 1, 23),
            std::make_tuple(1, 1, 15),
            std::make_tuple(1, 2, 10)
        };

        std::vector<std::tuple<size_t, size_t, int> > res;
        for (size_t i = 0; i < features.shape()[0]; ++i)
        {
            for (auto it = features.begin_instance_nonzero(i); it != features.end_instance_nonzero(i); ++it)
            {
                res.push_back(std::make_tuple(i, (*it).first, (*it).second));
            }
        }
        vigra_assert(res == expected, "Error in SparseFeatureGetter::ConstNonZeroIter.");
    }

    // Test iterator.
    {
        SparseFeatureGetter<int> features(Shape2(2, 3));
        features(0, 1) = 23;
        features(1, 1) = 15;
        features(1, 2) = 10;

        std::vector<std::vector<int> > expected {
            {0, 23, 0},
            {0, 15, 10}
        };

        std::vector<std::vector<int> > res(features.shape()[0]);
        for (size_t i = 0; i < features.shape()[0]; ++i)
        {
            for (auto it = features.begin_instance(i); it != features.end_instance(i); ++it)
            {
                res[i].push_back(*it);
            }
        }
        vigra_assert(res == expected, "Error in SparseFeatureGetter::ConstIter.");
    }

    // Test unsafe_insert.
    {
        SparseFeatureGetter<int> features(Shape2(2, 3));
        features.unsafe_insert(0, 1, 23);
        features.unsafe_insert(1, 1, 15);
        features.unsafe_insert(1, 2, 10);
        vigra_assert(features.count_nonzero() == 3, "Error in SparseFeatureGetter::unsafe_insert().");

        std::vector<int> expected {
            0, 23, 0, 0, 15, 10
        };

        std::vector<int> feats;
        for (size_t i = 0; i < features.shape()[0]; ++i)
        {
            for (size_t j = 0; j < features.shape()[1]; ++j)
            {
                feats.push_back(features(i, j));
            }
        }
        vigra_assert(feats == expected, "Error in SparseFeatureGetter::unsafe_insert().");
    }

    std::cout << "test_featuregetter(): Success!" << std::endl;
}

int main()
{
    test_featuregetter();
}
