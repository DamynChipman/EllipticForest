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

    // int nCoarse = 4;
    // int nFine = 8;
    // EllipticForest::InterpolationMatrixFine2Coarse<double> L21(nCoarse);
    // EllipticForest::InterpolationMatrixCoarse2Fine<double> L12(nFine);

    // std::cout << "L21 = " << L21 << std::endl;
    // std::cout << "L12 = " << L12 << std::endl;

    // Create uniform p4est
    int minLevel = 1;
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
    std::size_t nx, ny;
    if (argc > 1) {
        nx = (std::size_t) atoi(argv[1]);
        ny = (std::size_t) atoi(argv[1]);
    }
    else {
        nx = 4;
        ny = 4;
    }
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
        return x;
    });
    pde.setDUDY([](double x, double y){
        return y;
    });

    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, rootPatch, p4est);
    // EllipticForest::FISHPACK::FISHPACKQuadtree& quadtree = *HPS.quadtree;
    HPS.run();

    // Get merged root T
    EllipticForest::Matrix<double> T_merged = HPS.quadtree->data()[HPS.quadtree->globalIndices()[0][0]].T;

    // Create refined T
    EllipticForest::FISHPACK::FISHPACKFVGrid fineGrid(nx*2, ny*2, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKFVSolver solver;
    EllipticForest::Matrix<double> T_fine = solver.buildD2N(fineGrid);
    EllipticForest::Matrix<double> T_diff = T_merged - T_fine;

    double maxDiff = 0;
    for (auto i = 0; i < T_merged.nRows(); i++) {
        for (auto j = 0; j < T_merged.nCols(); j++) {
            double diff = T_merged(i,j) - T_fine(i,j);
            maxDiff = fmax(maxDiff, fabs(diff));
            printf("i = %4i,  j = %4i,  T_merged(i,j) = %12.4e,  T_fine(i,j) = %12.4e,  diff = %12.4e,  maxDiff = %12.4e\n", i, j, T_merged(i,j), T_fine(i,j), diff, maxDiff);
        }
    }

    double infNorm = EllipticForest::matrixInfNorm(T_merged, T_fine);
    printf("infNorm = %24.16e\n", infNorm);

    plt::matshow(T_fine, 1e-2);
    plt::title("T_fine");

    plt::matshow(T_merged, 1e-2);
    plt::title("T_merged");

    plt::matshow(T_diff, 1e-2);
    plt::title("T_diff");

    std::cout << "T_diff = " << T_diff << std::endl;

    plt::show();
    return EXIT_SUCCESS;
}