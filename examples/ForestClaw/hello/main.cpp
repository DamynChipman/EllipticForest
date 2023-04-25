#include <iostream>

#include <EllipticForest.hpp>
#include <ForestClaw.hpp>

#include <forestclaw2d.h>

int main(int argc, char** argv) {
    // Initialize ForestClaw
    fclaw_app_t* fclaw_app = fclaw_app_new(&argc, &argv, NULL);
    fclaw_global_essentialf("Hello from ForestClaw!");

    // Initialize EllipticForest
    EllipticForest::EllipticForestApp ef_app = EllipticForest::EllipticForestApp(&argc, &argv);
    ef_app.log("Hello from EllipticForest!");
    return EXIT_SUCCESS;
}