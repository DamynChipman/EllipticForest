#include "gtest/gtest.h"
#include <Patches/FiniteVolume/FiniteVolumePatch.hpp>

using namespace EllipticForest;

TEST(FiniteVolumePatch, init) {

    EXPECT_NO_THROW(FiniteVolumePatch patch1{});
    EXPECT_NO_THROW(FiniteVolumePatch patch2(MPI_COMM_WORLD));
    EXPECT_NO_THROW(FiniteVolumePatch patch2(MPI_COMM_WORLD, {}));

}

TEST(FiniteVolumePatch, data) {

    int nx = 4;
    double xlower = 0;
    double xupper = 1;
    int ny = 4;
    double ylower = 0;
    double yupper = 1;
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, xlower, xupper, ny, ylower, yupper);
    FiniteVolumePatch patch(MPI_COMM_WORLD, grid);

    EXPECT_EQ(patch.grid().nx(), nx);
    EXPECT_EQ(patch.matrixT().nRows(), 0);
    EXPECT_EQ(patch.vectorU().size(), 0);

}