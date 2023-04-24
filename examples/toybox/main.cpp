#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <EllipticForestApp.hpp>
#include <Quadtree.hpp>

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

std::vector<double> refineFunction(double& parentData) {
    std::vector<double> childrenData = {parentData/4.0, parentData/4.0, parentData/4.0, parentData/4.0};
    return childrenData;
}

double coarsenFunction(double& c0, double& c1, double& c2, double& c3) {
    return c0 + c1 + c2 + c3;
}

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.log("Hello, there!");
    // app.options.setFromFile(argv[1]);

    // Build quadtree
    std::cout << "Creating quadtree..." << std::endl;
    EllipticForest::Quadtree<double> quadtree{};
    quadtree.buildFromRoot(10.0);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Refining node 0..." << std::endl;
    quadtree.refineNode(0, refineFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Refining node 3..." << std::endl;
    quadtree.refineNode(3, refineFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Refining node 1..." << std::endl;
    quadtree.refineNode(1, refineFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Refining node 10..." << std::endl;
    quadtree.refineNode(10, refineFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Refining node 2..." << std::endl;
    quadtree.refineNode(2, refineFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    // std::cout << "Refining node 12..." << std::endl;
    // quadtree.refineNode(12, refineFunction);
    // std::cout << quadtree;
    // std::cout << "Data: ";
    // quadtree.traversePreOrder([&](double& data){
    //     std::cout << data << ", ";
    // });
    // std::cout << std::endl << std::endl;

    std::cout << "Coarsening node 14..." << std::endl;
    quadtree.coarsenNode(14, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    // quadtree.refineNode(1, refineFunction);
    // std::cout << quadtree << std::endl;

    // quadtree.refineNode(3, refineFunction);
    // std::cout << quadtree << std::endl;

    // quadtree.refineNode(13, refineFunction);
    // std::cout << quadtree << std::endl;

    // quadtree.refineNode(12, refineFunction);
    // std::cout << quadtree << std::endl;
    
    // // Get the options
    // int minLevel = std::get<int>(app.options["min-level"]);
    // int maxLevel = std::get<int>(app.options["max-level"]);
    // int nx = std::get<int>(app.options["nx"]);
    // int ny = std::get<int>(app.options["ny"]);
    // std::string mode = minLevel == maxLevel ? "uniform" : "adaptive";

    // // Create p4est
    // int fillUniform = 1;
    // int refineRecursive = 1;
    // p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(0, 1, 0, 1);
    // p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);
    // // p4est->user_pointer = &pde;

    // p4est_refine(p4est, refineRecursive,
    // [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

    //     // Get app context
    //     // auto& pde = *((EllipticForest::FISHPACK::FISHPACKProblem*) p4est->user_pointer);
    //     auto& app = EllipticForest::EllipticForestApp::getInstance();
    //     int maxLevel = std::get<int>(app.options["max-level"]);

    //     // Do not refine if at the max level
    //     if (quadrant->level >= maxLevel) {
    //         return 0;
    //     }

    //     // Refine once
    //     if (quadrant->level == 0) {
    //         return 1;
    //     }

    //     // Get bounds of quadrant
    //     double vxyz[3];
    //     double xLower, xUpper, yLower, yUpper;
    //     p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
    //     xLower = vxyz[0];
    //     yLower = vxyz[1];

    //     p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
    //     xUpper = vxyz[0];
    //     yUpper = vxyz[1];

    //     double epsilon = 0.1;
    //     double threshold = (1.0 - pow(2, -quadrant->level)) - epsilon;

    //     if (xLower > threshold && yUpper > threshold) {
    //         return 1;
    //     }
    //     else {
    //         return 0;
    //     }

    //     return 0;
    // },
    // NULL);

    // // Balance the p4est
    // p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);

    // // Save initial mesh
    // std::string VTKFilename = "toybox_mesh";
    // p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());

    // // Create quadtree
    // EllipticForest::Quadtree<double> quadtree(p4est);
    // quadtree.build(0.0, [&](double& parentNode, std::size_t childIndex){
    //     return parentNode + 1;
    // });

    // std::cout << quadtree << std::endl;

    return EXIT_SUCCESS;
}