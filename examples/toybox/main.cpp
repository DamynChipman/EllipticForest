#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <EllipticForestApp.hpp>
#include <Quadtree.hpp>

#include <EllipticForestApp.hpp>
#include <P4est.hpp>
#include <Quadtree.hpp>

std::vector<double> refineFunction(double& parentData) {
    std::vector<double> childrenData = {parentData/4.0, parentData/4.0, parentData/4.0, parentData/4.0};
    return childrenData;
}

double coarsenFunction(double& c0, double& c1, double& c2, double& c3) {
    return c0 + c1 + c2 + c3;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.log("Hello, there!");

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

    std::cout << "Coarsening node 14..." << std::endl;
    quadtree.coarsenNode(14, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Coarsening node 11..." << std::endl;
    quadtree.coarsenNode(11, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Coarsening node 2..." << std::endl;
    quadtree.coarsenNode(2, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Coarsening node 1..." << std::endl;
    quadtree.coarsenNode(1, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    std::cout << "Coarsening node 0..." << std::endl;
    quadtree.coarsenNode(0, coarsenFunction);
    std::cout << quadtree;
    std::cout << "Data: ";
    quadtree.traversePreOrder([&](double& data){
        std::cout << data << ", ";
    });
    std::cout << std::endl << std::endl;

    return EXIT_SUCCESS;
}