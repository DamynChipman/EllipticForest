#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <EllipticForest.hpp>
#include <Quadtree.hpp>

int main(int argc, char** argv) {

    // Create app
    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.logHead("Starting toybox...");

    // Create p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    int minLevel = 0;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);

    // Refine the p4est according to the RHS up to the max level
    p4est_refine(p4est, refineRecursive,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 3) {
            return 0;
        }
        else {
            return 1;
        }

    },
    NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);

    // Save initial mesh
    bool vtkFlag = true;
    if (vtkFlag) {
        std::string VTKFilename = "toybox_mesh";
        p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());
    }

    // Create leaf level root patch
    int nx = 16;
    int ny = 16;
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
    EllipticForest::HPSAlgorithm
        <EllipticForest::FISHPACK::FISHPACKFVGrid,
        EllipticForest::FISHPACK::FISHPACKFVSolver,
        EllipticForest::FISHPACK::FISHPACKPatch,
        double>
            HPS(rootPatch, solver);

    // 2. Call the setup stage
    HPS.setupStage(p4est);

    return EXIT_SUCCESS;
}