#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <MPI.hpp>
#include <FISHPACK.hpp>
#include <P4est.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>

namespace ef = EllipticForest;

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    // Create app
    ef::EllipticForestApp app(&argc, &argv);
    ef::MPI::MPIObject mpi(MPI_COMM_WORLD);

    app.logHead("Creating grid and patch");
    ef::FISHPACK::FISHPACKFVGrid grid;
    ef::FISHPACK::FISHPACKPatch patch;
    if (mpi.getRank() == ef::MPI::HEAD_RANK) {
        // Create grid
        int nx = 8;
        int ny = 14;
        double xlower = -1;
        double xupper = 1;
        double ylower = -2;
        double yupper = 3;
        grid = ef::FISHPACK::FISHPACKFVGrid(nx, ny, xlower, xupper, ylower, yupper);

        // Create patch
        patch = ef::FISHPACK::FISHPACKPatch(grid);
        patch.matrixT() = ef::Matrix<double>(2, 3, 4.22);
        patch.vectorF() = ef::Vector<double>({1, 2, 3, 4});
    }

    // Broadcast patch
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));
    app.log("patch = \n====================\n" + patch.str());
    ef::MPI::broadcast(patch, 0, mpi.getComm());

    // Print to console
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));
    app.log("patch = \n====================\n" + patch.str());

    return EXIT_SUCCESS;
}