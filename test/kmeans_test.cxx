#include <iostream>
#include <vector>

#include <vigra/multi_array.hxx>
#include <vigra/kmeans.hxx>

void test_kmeans()
{
    using namespace vigra;

    {
        typedef double FeatureType;

        std::vector<FeatureType> points_data {
            7.5, 1., 1.2, 1.5, 2., 2.5, 2.5, 3., 4., 4.5, 4.5, 5., 5.5, 5.5, 5.9, 6.0, 6.4, 6.5, 6.5, 7.5,
            1.8, 2.9, 3.5, 1.7, 2.5, 3.1, 1.5, 2.2, 5.5, 4.6, 6.2, 5.2, 6.1, 5.0, 1.9, 2.5, 1.9, 1.5, 2.6, 2.3
        };

        MultiArray<2, FeatureType> points(Shape2(20, 2));
        for (size_t i = 0; i < points_data.size(); ++i)
            points[i] = points_data[i];

        std::vector<size_t> instance_clusters;
        kmeans(points, 3, instance_clusters);

        for (size_t i = 0; i < points.shape()[0]; ++i)
        {
            std::cout << instance_clusters[i] << ": ";
            for (size_t j = 0; j < points.shape()[1]; ++j)
            {
                std::cout << points(i, j) << ", ";
            }
            std::cout << std::endl;
        }
    }
}

int main()
{
    test_kmeans();
}
