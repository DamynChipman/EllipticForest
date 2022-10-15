#include <iostream>
#include <utility>
#include <string>

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

    int nCoarse = 4;
    int nFine = 8;
    EllipticForest::InterpolationMatrixFine2Coarse<double> L21(nCoarse);
    EllipticForest::InterpolationMatrixCoarse2Fine<double> L12(nFine);

    std::cout << "L21 = " << L21 << std::endl;
    std::cout << "L12 = " << L12 << std::endl;

    // Create uniform p4est
    int minLevel = 2;
    int fillUniform = 1;
    p4est_connectivity_t* conn = p4est_connectivity_new_unitsquare();
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);

    // Write to VTK file to check
    std::string VTKFilename = "test_uniform";
    p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());

    // Create quadtree
    MyQuadtree quadtree(p4est);
    quadtree.build({0,0});

    std::cout << quadtree << std::endl;

    // quadtree.traversePreOrder([&](NodePair& nodePair) {
    //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // });

    // std::cout << "=====" << std::endl;

    // quadtree.traversePostOrder([&](NodePair& nodePair) {
    //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // });

    // Create root grid and patch
    std::size_t nx = 32;
    std::size_t ny = 32;
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
    
    // Create PDE to solve
    EllipticForest::FISHPACK::FISHPACKProblem pde;
    pde.setU([](double x, double y){
        return x*x + y*y;
    });
    pde.setF([](double x, double y){
        return 2.0;
    });
    pde.setDUDX([](double x, double y){
        return x;
    });
    pde.setDUDY([](double x, double y){
        return y;
    });

    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, rootPatch, p4est);
    HPS.run();

    return EXIT_SUCCESS;
}