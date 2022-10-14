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

    EllipticForest::FISHPACK::FISHPACKFVGrid grid(4, 4, -1, 1, -1, 1);
    EllipticForest::FISHPACK::FISHPACKPatch rootPatch;
    rootPatch.grid = grid;

    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(rootPatch, p4est);
    HPS.run();

    return EXIT_SUCCESS;
}