#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <EllipticForestApp.hpp>
// #include <HPSAlgorithm.hpp>
// #include <Quadtree.hpp>
// #include <PatchGrid.hpp>
// #include <PatchSolver.hpp>
// #include <Patch.hpp>
#include <FISHPACK.hpp>
#include <SpecialMatrices.hpp>
#include <p4est.h>
#include <p4est_connectivity.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

namespace plt = matplotlibcpp;
using NodePair = std::pair<std::size_t, std::size_t>;

class MyQuadtree : public EllipticForest::Quadtree<NodePair> {

public:

    MyQuadtree() {}
    MyQuadtree(p4est_t* p4est) : EllipticForest::Quadtree<NodePair>(p4est) {}

    NodePair initData(NodePair& parentData, std::size_t level, std::size_t index) {
        return {level, index};
    }

};

// class MyHPSAlgorithm : public EllipticForest::HomogeneousHPSMethod< {

// public:

//     MyHPSAlgorithm() {}

// protected:

//     void setupStage() const {

//         EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
//         app.log("Begin MyHPSAlgorithm-HPS Setup Stage");

        

//         app.log("End MyHPSAlgorithm-HPS Setup Stage");

//     }

// };

int main(int argc, char** argv) {
    std::cout << "Hello from toybox!" << std::endl;

    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Set options
    app.options.setOption("cache-operators", true);
    app.options.setOption("homogeneous-rhs", false);

    // CML options
    std::size_t nx = 4;
    std::size_t ny = 4;
    int minLevel = 2;
    if (argc > 1) {
        minLevel = atoi(argv[1]);
        nx = (std::size_t) atoi(argv[2]);
        ny = (std::size_t) atoi(argv[2]);
    }

    // int nCoarse = 4;
    // int nFine = 8;
    // EllipticForest::InterpolationMatrixFine2Coarse<double> L21(nCoarse);
    // EllipticForest::InterpolationMatrixCoarse2Fine<double> L12(nFine);

    // std::cout << "L21 = " << L21 << std::endl;
    // std::cout << "L12 = " << L12 << std::endl;

    // Create uniform p4est
    int fillUniform = 1;
    p4est_connectivity_t* conn = p4est_connectivity_new_unitsquare();
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);

    // Write to VTK file to check
    std::string VTKFilename = "test_uniform";
    p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());

    // Create quadtree
    // MyQuadtree quadtree(p4est);
    // quadtree.build({0,0});

    // std::cout << quadtree << std::endl;

    // quadtree.traversePreOrder([&](NodePair& nodePair) {
    //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // });

    // std::cout << "=====" << std::endl;

    // quadtree.traversePostOrder([&](NodePair& nodePair) {
    //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // });

    // Create root grid and patch
    double xLower = -1;
    double xUpper = 1;
    double yLower = -1;
    double yUpper = 1;
    EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKPatch rootPatch;
    rootPatch.grid = grid;
    rootPatch.level = 0;
    rootPatch.isLeaf = true;
    rootPatch.nCellsLeaf = nx;
    rootPatch.nPatchSideVector = {1, 1, 1, 1};
    
    // Create PDE to solve
    EllipticForest::FISHPACK::FISHPACKProblem pde;
    pde.setU([](double x, double y){
        return x*x + y*y;
    });
    pde.setF([](double x, double y){
        return 4.0;
    });
    pde.setDUDX([](double x, double y){
        return 2*x;
    });
    pde.setDUDY([](double x, double y){
        return 2*y;
    });
    // pde.setU([](double x, double y){
    //     return x + y;
    // });
    // pde.setF([](double x, double y){
    //     return 0.0;
    // });
    // pde.setDUDX([](double x, double y){
    //     return 1.0;
    // });
    // pde.setDUDY([](double x, double y){
    //     return 1.0;
    // });

    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, rootPatch, p4est);
    // EllipticForest::FISHPACK::FISHPACKQuadtree& quadtree = *HPS.quadtree;
    HPS.run();

    // Get merged root T
    // EllipticForest::Matrix<double> T_merged = HPS.quadtree->root().T;

    // // // Create refined T
    // EllipticForest::FISHPACK::FISHPACKFVGrid fineGrid(nx*4, ny*4, xLower, xUpper, yLower, yUpper);
    // EllipticForest::FISHPACK::FISHPACKFVSolver solver;
    // EllipticForest::Matrix<double> T_fine = solver.buildD2N(fineGrid);
    // EllipticForest::Matrix<double> T_diff = T_merged - T_fine;

    // double maxDiff = 0;
    // for (auto i = 0; i < T_merged.nRows(); i++) {
    //     for (auto j = 0; j < T_merged.nCols(); j++) {
    //         double diff = T_merged(i,j) - T_fine(i,j);
    //         maxDiff = fmax(maxDiff, fabs(diff));
    //         // printf("i = %4i,  j = %4i,  T_merged(i,j) = %12.4e,  T_fine(i,j) = %12.4e,  diff = %12.4e,  maxDiff = %12.4e\n", i, j, T_merged(i,j), T_fine(i,j), diff, maxDiff);
    //     }
    // }

    // double infNorm = EllipticForest::matrixInfNorm(T_merged, T_fine);
    // printf("Effective Resolution: [%i x %i]\n", HPS.quadtree->data()[0].grid.nPointsX(), HPS.quadtree->data()[0].grid.nPointsY());
    // printf("infNorm D2N Map = %24.16e\n", infNorm);

    // plt::matshow(T_fine, 1e-2);
    // plt::title("T_fine");

    // plt::matshow(T_merged, 1e-2);
    // plt::title("T_merged");

    // plt::matshow(T_diff, 1e-2);
    // plt::title("T_diff");

    // std::cout << "T_diff = " << T_diff << std::endl;

    // plt::show();

    // Compute error of solution
    double maxError = 0;
    HPS.quadtree->traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
        if (patch.isLeaf) {
            EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid;
            // std::cout << "PATCH: " << patch.ID << std::endl;
            for (auto i = 0; i < grid.nPointsX(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.nPointsY(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.nPointsY();
                    double diff = patch.u[index] - pde.u(x, y);
                    maxError = fmax(maxError, fabs(diff));
                    // printf("x = %12.4f,  y = %12.4f,  u_hps = %12.4f,  u_exact = %12.4f,  diff = %12.4e maxError = %12.4e\n", x, y, patch.u[index], pde.u(x,y), diff, maxError);
                }
                // std::cout << std::endl;
            }
        }
    });
    printf("infNorm Solution = %24.16e\n", maxError);


    return EXIT_SUCCESS;
}