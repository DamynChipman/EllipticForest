#include "Patches/FiniteVolume/FiniteVolumeGrid.hpp"

using namespace EllipticForest;

int rank, size;

void TEST_finite_volume_grid_init() {

    int nx = 16;
    int ny = 16;
    double x_lower = -1;
    double x_upper = 1;
    double y_lower = -1;
    double y_upper = 1;
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);

    DMView(grid.dm, 0);

}

int main(void) {

    MPI_Init(nullptr, nullptr);
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_finite_volume_grid_init();

    PetscFinalize();
    MPI_Finalize();

}