/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cmath>
#include <iostream>

#include <EllipticForest.hpp>

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Create p4est
    std::string filename = "/Users/damynchipman/packages/EllipticForest/data/meshes/hole.inp";
    p4est_connectivity_t* conn = p4est_connectivity_read_inp(filename.c_str());
    p4est_t* p4est = p4est_new(MPI_COMM_WORLD, conn, 0, NULL, NULL);

    // Refine the forest
    int refineLevel = 1;
    for (int level = 0; level < refineLevel; level++) {
        p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return 1;
        },
        NULL);
    }

    // Create leaf level root patch
    int nx = 8;
    int ny = 8;
    double xLower = -1;
    double xUpper = 1;
    double yLower = -1;
    double yUpper = 1;
    EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKPatch rootPatch(grid);
    rootPatch.level = 0;
    rootPatch.isLeaf = true;

    // Create patch solver
    EllipticForest::FISHPACK::FISHPACKFVSolver solver{};

    // Create and run HPS method
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm<EllipticForest::FISHPACK::FISHPACKFVGrid, EllipticForest::FISHPACK::FISHPACKFVSolver, EllipticForest::FISHPACK::FISHPACKPatch, double> HPS(rootPatch, solver);

    // 2. Call the setup stage
    HPS.setupStage(p4est);

    std::cout << HPS.quadtree << std::endl;

    // 3. Call the build stage
    HPS.buildStage();

    return EXIT_SUCCESS;
}