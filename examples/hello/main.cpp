#include <iostream>

#include <PlotUtils.hpp>
#include <EllipticForestApp.hpp>

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.log("Hello, there!");

#ifdef USE_MATPLOTLIBCPP
    app.logHead("matplotlibcpp enabled!");
#else
    app.logHead("matplotlibcpp not enabled...");
#endif

    return EXIT_SUCCESS;
}