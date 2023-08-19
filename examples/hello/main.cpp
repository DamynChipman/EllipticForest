/**
 * @file main.cpp : hello
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Checks installation of EllipticForest
 * 
 */

#include <iostream>

#include <EllipticForest.hpp>

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

/**
 * @brief Main driver for hello
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv) {

    // ====================================================
    // Create the app
    // ====================================================
    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.log("Hello, there!");

    // ====================================================
    // Check for matplotlibcpp
    // ====================================================
#ifdef USE_MATPLOTLIBCPP
    app.logHead("matplotlibcpp enabled!");
#else
    app.logHead("matplotlibcpp not enabled...");
#endif

    return EXIT_SUCCESS;
}