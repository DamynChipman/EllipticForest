#include "gtest/gtest.h"
#include <Patches/FiniteVolume/FiniteVolumeGrid.hpp>

using namespace EllipticForest;

TEST(FiniteVolumeGrid, init) {

    FiniteVolumeGrid empty_grid{};

    EXPECT_EQ(empty_grid.nx(), 0);
    EXPECT_EQ(empty_grid.ny(), 0);

    int nx = 8;
    double xlower = 0;
    double xupper = 1;
    int ny = 8;
    double ylower = 1;
    double yupper = 3;
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, xlower, xupper, ny, ylower, yupper);

    EXPECT_EQ(grid.nx(), nx);
    EXPECT_EQ(grid.xLower(), xlower);
    EXPECT_EQ(grid.xUpper(), xupper);
    EXPECT_EQ(grid.ny(), ny);
    EXPECT_EQ(grid.yLower(), ylower);
    EXPECT_EQ(grid.yUpper(), yupper);

}

TEST(FiniteVolumeGrid, points) {

    int nx = 4;
    double xlower = 0;
    double xupper = 1;
    int ny = 4;
    double ylower = 1;
    double yupper = 3;
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, xlower, xupper, ny, ylower, yupper);

    std::vector<double> expected_xpoints = {0.125, 0.375, 0.625, 0.875};
    std::vector<double> expected_ypoints = {1.25, 1.75, 2.25, 2.75};

    for (auto i = 0; i < nx; i++) {
        EXPECT_NEAR(grid.point(DimensionIndex::X, i), expected_xpoints[i], 1e-16);
    }
    for (auto j = 0; j < ny; j++) {
        EXPECT_NEAR(grid.point(DimensionIndex::Y, j), expected_ypoints[j], 1e-16);
    }

}