#include <iostream>
#include <vector>

#include <vigra/multi_array.hxx>
#include <vigra/kmeans.hxx>

void test_lineview()
{
    using namespace vigra;

    {
        typedef MultiArray<1, size_t> Array;
        Array x(50);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = i;
        std::vector<size_t> lines {
            5, 7, 10, 22, 29
        };
        detail::LineView<Array> view(x, lines);
        std::vector<size_t> x_out;
        for (auto it = view.begin(); it != view.end(); ++it)
            x_out.push_back(*it);
        vigra_assert(lines == x_out, "Error in LineView.");
    }

    {
        typedef MultiArray<2, double> Array;
        Array x(Shape2(5, 3));
        for (size_t j = 0; j < x.shape()[1]; ++j)
            for (size_t i = 0; i < x.shape()[0]; ++i)
                x(i, j) = (i+j)/2.;
        std::vector<size_t> lines {
            1, 3, 4
        };
        detail::LineView<Array> view(x, lines);
        std::vector<double> x_out;
        for (auto it = view.begin(); it != view.end(); ++it)
            x_out.push_back(*it);
        std::vector<double> x_expected {
            0.5, 1.5, 2.,
            1, 2., 2.5,
            1.5, 2.5, 3.
        };
        vigra_assert(x_out == x_expected, "Error in LineView.");
    }

}

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
    test_lineview();
    test_kmeans();
}
