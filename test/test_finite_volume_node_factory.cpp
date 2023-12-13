#include "gtest/gtest.h"
#include <Patches/FiniteVolume/FiniteVolumeNodeFactory.hpp>

using namespace EllipticForest;

TEST(FiniteVolumeNodeFactory, init) {

    EXPECT_NO_THROW(FiniteVolumeNodeFactory node_factory{});

}

TEST(FiniteVolumeNodeFactory, create_node) {

    FiniteVolumeNodeFactory node_factory{};

    int nx = 4;
    double xlower = 0;
    double xupper = 1;
    int ny = 4;
    double ylower = 0;
    double yupper = 1;
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, xlower, xupper, ny, ylower, yupper);
    FiniteVolumePatch patch(MPI_COMM_WORLD, grid);

    Node<FiniteVolumePatch>* node = node_factory.createNode(patch, "0", 0, 0, 0);

    EXPECT_EQ(node->data.grid().nx(), nx);
    EXPECT_EQ(node->data.grid().ny(), ny);
    EXPECT_EQ(node->path, "0");

    EXPECT_NO_FATAL_FAILURE(delete node);

}