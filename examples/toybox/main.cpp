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
#include <Quadtree.hpp>
#include <MPI.hpp>
#include <P4est.hpp>

int main(int argc, char** argv) {

    // Kokkos::initialize(argc, argv);
    EllipticForest::EllipticForestApp app{&argc, &argv};
    EllipticForest::MPI::MPIObject mpi{};
    app.logHead("Hello from toybox!");

    // Kokkos::finalize();
    return EXIT_SUCCESS;
}