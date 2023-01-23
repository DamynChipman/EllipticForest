#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <PlotUtils.hpp>
#include <EllipticForestApp.hpp>

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
    app.log("PETSc not enabled...");
#endif

    return EXIT_SUCCESS;
}