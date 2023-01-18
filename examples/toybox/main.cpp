#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <EllipticForestApp.hpp>
#include <P4est.hpp>
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

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.log("Hello, there!");
#ifdef USE_MATPLOTLIBCPP
    app.log("matplotlibcpp enabled!");
#else
    app.log("matplotlibcpp not enabled...");
#endif
#ifdef USE_PETSC
    app.log("PETSc enabled!");
#else
    app.log("PETSc not enabled...")
#endif

    return EXIT_SUCCESS;
}