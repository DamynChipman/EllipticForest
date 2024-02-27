#include "gtest/gtest.h"

#if USE_MATPLOTLIBCPP
#include "PlotUtils.hpp"

TEST(Plotting, meshgrid) {
    int M = 3;
    int N = 2;
    EllipticForest::Vector<double> x1 = {1.0, 2.0, 3.0};
    EllipticForest::Vector<double> x2 = {4.0, 5.0};

    auto [x1_mesh_test, x2_mesh_test] = matplotlibcpp::meshgrid(x1, x2);

    EllipticForest::Vector<double> x1_mesh_true = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    EllipticForest::Vector<double> x2_mesh_true = {4.0, 5.0, 4.0, 5.0, 4.0, 5.0};

    for (auto i = 0; i < M*N; i++) {
        EXPECT_EQ(x1_mesh_test[i], x1_mesh_true[i]);
        EXPECT_EQ(x2_mesh_test[i], x2_mesh_true[i]);
    }
}

#endif