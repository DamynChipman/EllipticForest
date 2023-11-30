#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <petsc.h>
#include <petscvec.h>
// #include <Kokkos_Core.hpp>
// #include <Kokkos_UnorderedMap.hpp>

#include <EllipticForestApp.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>
#include <P4est.hpp>

/**
 * @brief User-defined solution function
 * 
 * When computing error, calls this function to compare with numerical solution.
 * 
 * Used by EllipticForest to create the boundary data.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double uFunction(double x, double y) {
    return sin(x) + sin(y);
}

/**
 * @brief User-defined load function
 * 
 * Right-hand side function of the elliptic PDE.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double fFunction(double x, double y) {
    return -uFunction(x, y);
}

int main(int argc, char** argv) {

    // Kokkos::initialize(argc, argv);
    EllipticForest::EllipticForestApp app{&argc, &argv};
    EllipticForest::MPI::MPIObject mpi{};

    // Options
    int nx = 4;
    int ny = 4;
    double x_lower = -1;
    double x_upper = 1;
    double y_lower = -1;
    double y_upper = 1;
    int min_level = 2;
    int max_level = 2;
    double threshold = 1.2;


    // Create grid and patch prototypes
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // Create node factory and mesh
    EllipticForest::FiniteVolumeNodeFactory node_factory(MPI_COMM_WORLD);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            double f = fFunction(x, y);
            return fabs(f) > threshold;
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        node_factory
    );

    mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        app.log(EllipticForest::MPI::communicatorGetName(node->node_comm));
        return 1;
    });

    // Kokkos::finalize();
    return EXIT_SUCCESS;
}