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

namespace plt = matplotlibcpp;
using NodePair = std::pair<std::size_t, std::size_t>;

double f(double x) {
    return sin(4.0*M_PI*x);
}

void testInterpolationMatrices() {

    // Create interpolation matrices
    int nFine = 16;
    int nCoarse = 8;
    EllipticForest::InterpolationMatrixFine2Coarse<double> L21(nCoarse);
    EllipticForest::InterpolationMatrixCoarse2Fine<double> L12(nFine);

    // Create grids
    double xL = -1.0;
    double xU = 1.0;
    double yL = -1.0;
    double yU = 1.0;
    EllipticForest::FISHPACK::FISHPACKFVGrid fineGrid(nFine, nFine, xL, xU, yL, yU);
    EllipticForest::FISHPACK::FISHPACKFVGrid coarseGrid(nCoarse, nCoarse, xL, xU, yL, yU);

    EllipticForest::Vector<double> fFine(nFine);
    EllipticForest::Vector<double> xFine(nFine);
    for (auto i = 0; i < nFine; i++) {
        double x = fineGrid(XDIM, i);
        xFine[i] = x;
        fFine[i] = f(x);
    }

    EllipticForest::Vector<double> fCoarse(nCoarse);
    EllipticForest::Vector<double> xCoarse(nCoarse);
    for (auto i = 0; i < nCoarse; i++) {
        double x = coarseGrid(XDIM, i);
        xCoarse[i] = x;
        fCoarse[i] = f(x);
    }

    EllipticForest::Vector<double> fFine2Coarse = L21 * fFine;
    EllipticForest::Vector<double> fCoarse2Fine = L12 * fCoarse;

    // plt::figure();
    plt::named_plot("fine", xFine.data(), fFine.data(), "-o");
    plt::named_plot("coarse", xCoarse.data(), fCoarse.data(), "-o");
    plt::named_plot("fine2coarse", xCoarse.data(), fFine2Coarse.data(), "-o");
    plt::named_plot("coarse2fine", xFine.data(), fCoarse2Fine.data(), "-o");
    plt::legend();
    plt::show();

    EllipticForest::Vector<double> xFine4 = EllipticForest::concatenate({xFine, xFine, xFine, xFine});
    EllipticForest::Vector<double> xCoarse4 = EllipticForest::concatenate({xCoarse, xCoarse, xCoarse, xCoarse});
    EllipticForest::Vector<double> fFine4 = EllipticForest::concatenate({fFine, fFine, fFine, fFine});
    EllipticForest::Vector<double> fCoarse4 = EllipticForest::concatenate({fCoarse, fCoarse, fCoarse, fCoarse});

    std::vector<EllipticForest::Matrix<double>> L21Diagonals = {L21, L21, L21, L21};
    std::vector<EllipticForest::Matrix<double>> L12Diagonals = {L12, L12, L12, L12};
    EllipticForest::Matrix<double> L21Block = EllipticForest::blockDiagonalMatrix(L21Diagonals);
    EllipticForest::Matrix<double> L12Block = EllipticForest::blockDiagonalMatrix(L12Diagonals);

    EllipticForest::Vector<double> fFine2Coarse4 = L21Block * fFine4;
    EllipticForest::Vector<double> fCoarse2Fine4 = L12Block * fCoarse4;

    // plt::figure();
    plt::named_plot("fine", xFine4.data(), fFine4.data(), "-o");
    plt::named_plot("coarse", xCoarse4.data(), fCoarse4.data(), "-o");
    plt::named_plot("fine2coarse", xCoarse4.data(), fFine2Coarse4.data(), "-o");
    plt::named_plot("coarse2fine", xFine4.data(), fCoarse2Fine4.data(), "-o");
    plt::legend();
    plt::show();

    std::cout << "L21Block = " << L21Block << std::endl;
    std::cout << "L12Block = " << L12Block << std::endl;



}

// class PolarStarPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

// public:

//     PolarStarPoissonProblem() {}

// private:

//     double hsmooth(int ID, double r, double theta);
//      hsmoothGrad(int ID, double r, double theta)

// };

// class MyQuadtree : public EllipticForest::Quadtree<NodePair> {

// public:

//     MyQuadtree() {}
//     MyQuadtree(p4est_t* p4est) : EllipticForest::Quadtree<NodePair>(p4est) {}

//     NodePair initData(NodePair& parentData, std::size_t level, std::size_t index) {
//         return {level, index};
//     }

// };

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

// int refine_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
//     if (quadrant->level < 4) {
//         return 1;
//     }
//     else {
//         return 0;
//         }
// }

struct Errors {
    Errors() {}
    int nDOFs;
    double D2NError;
    double l2Error;
    double lIError;
};

Errors solvePoissonViaHPS(EllipticForest::FISHPACK::FISHPACKProblem& pde) {

    // Get the options
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    int minLevel = std::get<int>(app.options["min-level"]);
    int maxLevel = std::get<int>(app.options["max-level"]);
    int nx = std::get<int>(app.options["nx"]);
    int ny = std::get<int>(app.options["ny"]);

    // Create uniform p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);
    p4est->user_pointer = &pde;

    p4est_refine(p4est, refineRecursive,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        // Get app context
        auto& pde = *((EllipticForest::FISHPACK::FISHPACKProblem*) p4est->user_pointer);
        auto& app = EllipticForest::EllipticForestApp::getInstance();
        int maxLevel = std::get<int>(app.options["max-level"]);
        double threshold = std::get<double>(app.options["refinement-threshold"]);

        // Do not refine if at the max level
        if (quadrant->level >= maxLevel) {
            return 0;
        }

        // Get midpoint of quadrant
        p4est_qcoord_t halfQuad = P4EST_QUADRANT_LEN(quadrant->level) / 2;
        double vxyz[3];
        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + halfQuad, quadrant->y + halfQuad, vxyz);
        double x = vxyz[0];
        double y = vxyz[1];

        // Refine by RHS value
        if (fabs(pde.f(x,y)) > threshold) {
            return 1;
        }
        else {
            return 0;
        }

        // Refine middle
        // double xRefineLower = -0.5;
        // double xRefineUpper = 0.5;
        // double yRefineLower = -0.5;
        // double yRefineUpper = 0.5;
        // if ((x >= xRefineLower && x <= xRefineUpper) && (y >= yRefineLower && y <= yRefineUpper)) {
        //     return 1;
        // }
        // else {
        //     return 0;
        // }

        return 0;
    },
    NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);

    std::string VTKFilename = "toybox_mesh";
    p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());

    // Create leaf level root patch
    double xLower = -1;
    double xUpper = 1;
    double yLower = -1;
    double yUpper = 1;
    EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKPatch leafPatch;
    leafPatch.grid = grid;
    leafPatch.globalID = 0;
    leafPatch.level = 0;
    leafPatch.isLeaf = true;
    // leafPatch.nPatchSideVector = {1, 1, 1, 1};

    // Create and run HPS method
    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, leafPatch, p4est);
    // std::cout << *HPS.quadtree << std::endl;
    HPS.run();
    HPS.toVTK("toybox");

    // Get merged root T
    // EllipticForest::Matrix<double> T_merged = HPS.quadtree->root().T;

    // // Create refined T
    // EllipticForest::FISHPACK::FISHPACKFVSolver solver;
    // EllipticForest::Matrix<double> T_fine = solver.buildD2N(HPS.quadtree->root().grid);
    // EllipticForest::Matrix<double> T_diff = T_merged - T_fine;

    // double maxDiff = 0;
    // for (auto i = 0; i < T_merged.nRows(); i++) {
    //     for (auto j = 0; j < T_merged.nCols(); j++) {
    //         double diff = T_merged(i,j) - T_fine(i,j);
    //         maxDiff = fmax(maxDiff, fabs(diff));
    //     }
    // }

    // double infNorm = EllipticForest::matrixInfNorm(T_merged, T_fine);
    // int resolution = HPS.quadtree->data()[0].grid.nPointsX();
    // // printf("Effective Resolution: [%i x %i]\n", resolution, resolution);
    // printf("infNorm D2N Map = %24.16e\n", infNorm);

    // Compute error of solution
    double maxError = 0;
    HPS.quadtree->traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
        if (patch.isLeaf) {
            EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid;
            for (auto i = 0; i < grid.nPointsX(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.nPointsY(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.nPointsY();
                    int index_T = i + j*grid.nPointsY();
                    double diff = patch.u[index_T] - pde.u(x, y);
                    maxError = fmax(maxError, fabs(diff));
                }
            }
        }
    });
    int resolution = pow(2,maxLevel)*nx;
    app.log("infNorm Solution = %24.16e", maxError);
    app.log("Effective resolution = [%i,%i]", resolution, resolution);

    Errors err;
    err.nDOFs = resolution;
    // err.D2NError = infNorm;
    err.lIError = maxError;
    return err;

}

int main(int argc, char** argv) {
    std::cout << "Hello from toybox!" << std::endl;

    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Set options
    app.options.setOption("cache-operators", false);
    app.options.setOption("homogeneous-rhs", false);

    // testInterpolationMatrices();

    // CML options
    std::size_t nx = 4;
    std::size_t ny = 4;
    int minLevel = 1;
    int maxLevel = 3;
    double refinementThreshold = 1;
    if (argc > 1) {
        minLevel = atoi(argv[1]);
        maxLevel = atoi(argv[2]);
        nx = (std::size_t) atoi(argv[3]);
        ny = (std::size_t) atoi(argv[3]);
        refinementThreshold = atof(argv[4]);
    }
    app.options.setOption("min-level", minLevel);
    app.options.setOption("max-level", maxLevel);
    app.options.setOption("nx", (int) nx);
    app.options.setOption("ny", (int) ny);
    app.options.setOption("refinement-threshold", refinementThreshold);

    // Create PDE to solve
    EllipticForest::FISHPACK::FISHPACKProblem pde;
    // pde.setU([](double x, double y){
    //     return 0.25*(pow(x,2) + pow(y,2));
    // });
    // pde.setF([](double x, double y){
    //     return 1.0;
    // });
    // pde.setU([](double x, double y){
    //     return (1.0/6.0)*(pow(x,3) + 2.0*pow(y,3));
    // });
    // pde.setF([](double x, double y){
    //     return x + 2.0*y;
    // });
    // pde.setDUDX([](double x, double y){
    //     return pow(x,2) / 2.0;
    // });
    // pde.setDUDY([](double x, double y){
    //     return pow(y,2);
    // });
    double x0 = 0.5;
    double y0 = 0.5;
    pde.setU([&](double x, double y){
        return exp(-12.5*pow(x-x0,2) - 12.5*pow(y-y0,2));
    });
    pde.setF([&](double x, double y){
        return (-50.0 + 625.0*pow(x-x0,2) + 625.0*pow(y-y0,2))*exp(-12.5*pow(x-x0,2) - 12.5*pow(y-y0,2));
    });
    // pde.setU([&](double x, double y){
    //     return exp(-12.5*pow(x-x0,2));
    // });
    // pde.setF([&](double x, double y){
    //     return -25.0*exp(-12.5*pow(x-x0,2)) + 625.0*exp(-12.5*pow(x-x0,2))*pow(x-x0,2);
    // });
    // pde.setDUDX([](double x, double y){
    //     return 0;
    // });
    // pde.setDUDY([](double x, double y){
    //     return 0;
    // });
    // pde.setU([](double x, double y){
    //     // return x + y;
    //     double pi23 = M_PI * (2.0/3.0);
    //     return sin(pi23*x)*sinh(pi23*y);
    // });
    // pde.setF([](double x, double y){
    //     return 0.0;
    // });
    // pde.setDUDX([](double x, double y){
    //     return 2.0*x + 2.0*pow(y,3);
    // });
    // pde.setDUDY([](double x, double y){
    //     return 2.0*y + 6.0*x*pow(y,2);
    // });
    // pde.setU([](double x, double y){
    //     return sin(4.0*M_PI*x) + cos(4.0*M_PI*y);
    // });
    // pde.setF([](double x, double y){
    //     return -16.0*pow(M_PI,2)*(cos(4.0*M_PI*y) + sin(4.0*M_PI*x));
    // });
    // pde.setDUDX([](double x, double y){
    //     return 4.0*M_PI*cos(4.0*M_PI*x);
    // });
    // pde.setDUDY([](double x, double y){
    //     return -4.0*M_PI*sin(4.0*M_PI*y);
    // });

    if (argc > 1) {
        auto err = solvePoissonViaHPS(pde);
        return EXIT_SUCCESS;
    }

    std::vector<int> leafPatchSizeVector = {4, 8, 16, 32, 64, 128};
    // std::vector<int> leafPatchSizeVector = {4, 8, 16, 32};
    std::vector<int> levelVector = {0, 1, 2, 3, 4};
    std::vector<int> nDOFsVector;
    std::vector<double> D2NErrorVector;
    std::vector<double> uErrorVector;
    std::vector<double> buildTimeVector;
    std::vector<double> upwardsTimeVector;
    std::vector<double> solveTimeVector;

    std::vector<std::vector<int>> nDOFsPlots;
    std::vector<std::vector<double>> solutionPlots;
    std::vector<std::vector<double>> D2NPlots;
    std::vector<std::vector<double>> buildTimePlots;
    std::vector<std::vector<double>> upwardsTimePlots;
    std::vector<std::vector<double>> solveTimePlots;

    for (auto& leafPatchSize : leafPatchSizeVector) {
        
        for (auto& level : levelVector) {
            // Set the options
            app.options.setOption("min-level", 1);
            app.options.setOption("max-level", level);
            app.options.setOption("nx", leafPatchSize);
            app.options.setOption("ny", leafPatchSize);

            Errors err = solvePoissonViaHPS(pde);
            nDOFsVector.push_back(err.nDOFs);
            D2NErrorVector.push_back(err.D2NError);
            uErrorVector.push_back(err.lIError);

            buildTimeVector.push_back(app.timers["build-stage"].time());
            upwardsTimeVector.push_back(app.timers["upwards-stage"].time());
            solveTimeVector.push_back(app.timers["solve-stage"].time());

            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();

            app.log("nDOFs = %i", err.nDOFs);
            app.log("error = %24.16e", err.lIError);
        }

        // plt::named_loglog("Solution: N = " + std::to_string(leafPatchSize), nDOFsVector, uErrorVector, "-*");
        // plt::named_loglog("D2N Map: N = " + std::to_string(leafPatchSize), nDOFsVector, D2NErrorVector, "--v");

        // plt::named_loglog("Build Time: N = " + std::to_string(leafPatchSize), nDOFsVector, buildTimeVector, "-*");
        plt::named_loglog("Upwards Time: N = " + std::to_string(leafPatchSize), nDOFsVector, upwardsTimeVector, "-*");
        // plt::named_loglog("Solve Time: N = " + std::to_string(leafPatchSize), nDOFsVector, solveTimeVector, "-*");

        // nDOFsPlots.push_back(nDOFsVector);
        // solutionPlots.push_back(uErrorVector);
        // D2NPlots.push_back(D2NErrorVector);
        // buildTimePlots.push_back(buildTimeVector);
        // upwardsTimePlots.push_back(upwardsTimeVector);
        // solveTimePlots.push_back(solveTimeVector);

        nDOFsVector.clear();
        D2NErrorVector.clear();
        uErrorVector.clear();
        buildTimeVector.clear();
        upwardsTimeVector.clear();
        solveTimeVector.clear();

    }

    std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<std::string> xTickLabels;
    for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
    plt::xlabel("Total Grid Resolution");
    plt::ylabel("Time [sec]");
    // plt::ylabel("Inf-Norm Error");
    plt::title("Adaptive Grid - Timing");
    plt::xticks(xTicks, xTickLabels);
    plt::legend();
    plt::grid(true);
    plt::save("plot_adaptive_upwards_time.pdf");


    // Create plots
    // std::vector<std::string> titles = {
    //     "Adaptive Grid Convergence - Solution",
    //     "Adaptive Grid Convergence - Dirichlet-to-Neumann Map",
    //     "Adaptive Grid Convergence - Build Time",
    //     "Adaptive Grid Convergence - Upwards Time",
    //     "Adaptive Grid Convergence - Solve Time"
    // };
    // std::vector<std::vector<std::vector<double>>> plots = {
    //     solutionPlots,
    //     D2NPlots,
    //     buildTimePlots,
    //     upwardsTimePlots,
    //     solutionPlots
    // };
    // std::vector<std::string> names = {
    //     "plot_Adaptive_convergence.pdf",
    //     "plot_Adaptive_D2N.pdf",
    //     "plot_Adaptive_build_time.pdf",
    //     "plot_Adaptive_upwards_time.pdf",
    //     "plot_Adaptive_solve_time.pdf"
    // };
    // for (auto i = 0; i < 5; i++) {

    //     for (auto& p : plots[i]) {
    //         plt::named_loglog("N = " + std::to_string(leafPatchSizeVector[i]), nDOFsPlots[i], p, "-*");
    //     }

    //     std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    //     std::vector<std::string> xTickLabels;
    //     for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
    //     plt::xlabel("Total Grid Resolution");
    //     plt::ylabel("L-Infinity Norm");
    //     plt::title(titles[i]);
    //     plt::xticks(xTicks, xTickLabels);
    //     plt::legend();
    //     plt::grid(true);
    //     plt::save(names[i]);

    // }

    // plt::xlabel("Total Grid Resolution");
    // plt::ylabel("Time [sec]");
    // plt::title("Uniform Grid Timing - Solve Stage");
    // plt::xticks(xTicks, xTickLabels);
    // plt::legend();
    // plt::grid(true);
    // plt::save("plot_uniform_timing_solve.pdf");
    

    // // int nCoarse = 4;
    // // int nFine = 8;
    // // EllipticForest::InterpolationMatrixFine2Coarse<double> L21(nCoarse);
    // // EllipticForest::InterpolationMatrixCoarse2Fine<double> L12(nFine);

    // // std::cout << "L21 = " << L21 << std::endl;
    // // std::cout << "L12 = " << L12 << std::endl;

    // // Create uniform p4est
    // int fillUniform = 1;
    // int refineRecursive = 1;
    // p4est_connectivity_t* conn = p4est_connectivity_new_unitsquare();
    // p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);
    // p4est->user_pointer = &pde;

    // p4est_refine(p4est, refineRecursive,
    // [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     auto& pde = *((EllipticForest::FISHPACK::FISHPACKProblem*) p4est->user_pointer);
    //     auto& app = EllipticForest::EllipticForestApp::getInstance();
    //     int maxLevel = std::get<int>(app.options["max-level"]);

    //     p4est_qcoord_t xQCoord, yQCoord;
    //     double vxyz[3];
    //     p4est_qcoord_to_vertex(p4est->connectivity, which_tree, xQCoord, yQCoord, vxyz);
    //     double x = vxyz[0];
    //     double y = vxyz[1];

    //     if (quadrant->level >= maxLevel) {
    //         return 0;
    //     }

    //     if (fabs(pde.f(x,y)) > 200.0) {
    //         return 1;
    //     }
    //     else {
    //         return 0;
    //     }
    // },
    // NULL);

    // // Write to VTK file to check
    // std::string VTKFilename = "test_uniform";
    // p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());

    // // Create quadtree
    // // MyQuadtree quadtree(p4est);
    // // quadtree.build({0,0});

    // // std::cout << quadtree << std::endl;

    // // quadtree.traversePreOrder([&](NodePair& nodePair) {
    // //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // // });

    // // std::cout << "=====" << std::endl;

    // // quadtree.traversePostOrder([&](NodePair& nodePair) {
    // //     std::cout << nodePair.first << ",  " << nodePair.second << std::endl;
    // // });

    // // Create root grid and patch
    // double xLower = -1;
    // double xUpper = 1;
    // double yLower = -1;
    // double yUpper = 1;
    // EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    // EllipticForest::FISHPACK::FISHPACKPatch leafPatch;
    // leafPatch.grid = grid;
    // leafPatch.level = 0;
    // leafPatch.isLeaf = true;
    // leafPatch.nCellsLeaf = nx;
    // leafPatch.nPatchSideVector = {1, 1, 1, 1};
    
    // // pde.setU([](double x, double y){
    // //     return x + y;
    // // });
    // // pde.setF([](double x, double y){
    // //     return 0.0;
    // // });
    // // pde.setDUDX([](double x, double y){
    // //     return 1.0;
    // // });
    // // pde.setDUDY([](double x, double y){
    // //     return 1.0;
    // // });

    // EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, leafPatch, p4est);
    // // EllipticForest::FISHPACK::FISHPACKQuadtree& quadtree = *HPS.quadtree;
    // HPS.run();

    // // Get merged root T
    // EllipticForest::Matrix<double> T_merged = HPS.quadtree->root().T;

    // // Create refined T
    // // EllipticForest::FISHPACK::FISHPACKFVGrid fineGrid(nx*4, ny*4, xLower, xUpper, yLower, yUpper);
    // EllipticForest::FISHPACK::FISHPACKFVSolver solver;
    // EllipticForest::Matrix<double> T_fine = solver.buildD2N(HPS.quadtree->root().grid);
    // EllipticForest::Matrix<double> T_diff = T_merged - T_fine;

    // double maxDiff = 0;
    // for (auto i = 0; i < T_merged.nRows(); i++) {
    //     for (auto j = 0; j < T_merged.nCols(); j++) {
    //         double diff = T_merged(i,j) - T_fine(i,j);
    //         maxDiff = fmax(maxDiff, fabs(diff));
    //         // printf("i = %4i,  j = %4i,  T_merged(i,j) = %12.4e,  T_fine(i,j) = %12.4e,  diff = %12.4e,  maxDiff = %12.4e\n", i, j, T_merged(i,j), T_fine(i,j), diff, maxDiff);
    //     }
    // }

    // double infNorm = EllipticForest::matrixInfNorm(T_merged, T_fine);
    // printf("Effective Resolution: [%i x %i]\n", HPS.quadtree->data()[0].grid.nPointsX(), HPS.quadtree->data()[0].grid.nPointsY());
    // printf("infNorm D2N Map = %24.16e\n", infNorm);

    // // plt::matshow(T_fine, 1e-2);
    // // plt::title("T_fine");

    // // plt::matshow(T_merged, 1e-2);
    // // plt::title("T_merged");

    // // plt::matshow(T_diff, 1e-2);
    // // plt::title("T_diff");

    // // std::cout << "T_diff = " << T_diff << std::endl;

    // // plt::show();

    // // Compute error of solution
    // double maxError = 0;
    // HPS.quadtree->traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
    //     if (patch.isLeaf) {
    //         EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid;
    //         // std::cout << "PATCH: " << patch.ID << std::endl;
    //         for (auto i = 0; i < grid.nPointsX(); i++) {
    //             double x = grid(XDIM, i);
    //             for (auto j = 0; j < grid.nPointsY(); j++) {
    //                 double y = grid(YDIM, j);
    //                 int index = j + i*grid.nPointsY();
    //                 double diff = patch.u[index] - pde.u(x, y);
    //                 maxError = fmax(maxError, fabs(diff));
    //                 // printf("x = %12.4f,  y = %12.4f,  u_hps = %12.4f,  u_exact = %12.4f,  diff = %12.4e maxError = %12.4e\n", x, y, patch.u[index], pde.u(x,y), diff, maxError);
    //             }
    //             // std::cout << std::endl;
    //         }
    //     }
    // });
    // printf("infNorm Solution = %24.16e\n", maxError);


    return EXIT_SUCCESS;
}