#include <iostream>

#include <EllipticForest.hpp>

#include "common.hpp"

class PoissonProblem2 : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    int nPolar;
    std::vector<double> x0s;
    std::vector<double> y0s;
    std::vector<double> r0s;
    std::vector<double> r1s;
    std::vector<double> ns;
    double eps_disk = 0.015625;

    PoissonProblem2() {
        nPolar = 1;
        x0s = {0};
        y0s = {0};
        r0s = {0.4};
        r1s = {0.45};
        ns = {4};
    }

    PoissonProblem2(int nPolar, std::vector<double> x0s, std::vector<double> y0s, std::vector<double> r0s, std::vector<double> r1s, std::vector<double> ns, double epsilon) :
        nPolar(nPolar),
        x0s(x0s),
        y0s(y0s),
        r0s(r0s),
        r1s(r1s),
        ns(ns),
        eps_disk(epsilon)
            {}

    std::string name() override { return "poisson-2"; }

    double lambda() { return 0.0; }

    double u(double x, double y) override {
        double res = 0;
        for (auto i = 0; i < nPolar; i++) {
            double x0 = x0s[i];
            double y0 = y0s[i];
            double r = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
            double theta = atan2(y - y0, x - x0);
            res += 1.0 - hsmooth(i, r, theta);
        }
        return res;
    }

    double f(double x, double y) override {
        double res = 0;
        for (auto i = 0; i < nPolar; i++) {
            double x0 = x0s[i];
            double y0 = y0s[i];
            double r = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
            double theta = atan2(y - y0, x - x0);
            res -= hsmooth_laplacian(i, r, theta);
        }
        return res;
    }

    double dudx(double x, double y) override {
        return 0;
    }

    double dudy(double x, double y) override {
        return 0;
    }

private:

    double sech(double x) {
        return 1.0 / cosh(x);
    }

    void polar_interface_complete(int ID, double theta, double& p, double& dpdtheta, double& d2pdtheta2) {
        double r0 = r0s[ID];
        double r1 = r1s[ID];
        int n = ns[ID];

        p = r0*(1.0 + r1*cos(n*theta));
        dpdtheta = r0*(-n*r1*sin(n*theta));
        d2pdtheta2 = r0*(-pow(n,2)*r1*cos(n*theta));
    }

    double polar_interface(int ID, double theta) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);
        return p;
    }

    double hsmooth(int ID, double r, double theta) {
        double p = polar_interface(ID, theta);
        return (tanh((r - p)/eps_disk) + 1.0) / 2.0;
    }

    void hsmooth_grad(int ID, double r, double theta, double& grad_x, double& grad_y) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);
        
        double eps_disk2 = pow(eps_disk, 2);
        double sech2 = pow(sech((r - p)/eps_disk), 2);
        grad_x = sech2 / eps_disk2;
        grad_y = -dpdtheta*sech2/(eps_disk2*r);

    }

    double hsmooth_laplacian(int ID, double r, double theta) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);

        double eps_disk2 = pow(eps_disk, 2);
        double sech2 = pow(sech((r-p)/eps_disk), 2);
        double t = tanh((r-p)/eps_disk);
        double st = t*sech2;
        double s1 = pow(dpdtheta,2)*st/eps_disk2;
        double s2 = d2pdtheta2*sech2/(2*eps_disk);
        double s3 = st/eps_disk2;
        double s4 = sech2/(2*eps_disk*r);
        return (-s1-s2)/pow(r, 2) - s3 + s4;
    }

};

ResultsData run(EllipticForest::FISHPACK::FISHPACKProblem& pde) {

    // Get the options
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    int minLevel = std::get<int>(app.options["min-level"]);
    int maxLevel = std::get<int>(app.options["max-level"]);
    int nx = std::get<int>(app.options["nx"]);
    int ny = std::get<int>(app.options["ny"]);
    std::string mode = minLevel == maxLevel ? "uniform" : "adaptive";

    // Create p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);
    p4est->user_pointer = &pde;

    // Refine the p4est according to the RHS up to the max level
    p4est_refine(p4est, refineRecursive,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        // Get app context
        auto& pde = *((EllipticForest::FISHPACK::FISHPACKProblem*) p4est->user_pointer);
        auto& app = EllipticForest::EllipticForestApp::getInstance();
        int maxLevel = std::get<int>(app.options["max-level"]);
        int nx = std::get<int>(app.options["nx"]);
        int ny = std::get<int>(app.options["ny"]);
        double threshold = std::get<double>(app.options["refinement-threshold"]);

        // Do not refine if at the max level
        if (quadrant->level >= maxLevel) {
            return 0;
        }

        // Get bounds of quadrant
        double vxyz[3];
        double xLower, xUpper, yLower, yUpper;
        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
        xLower = vxyz[0];
        yLower = vxyz[1];

        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
        xUpper = vxyz[0];
        yUpper = vxyz[1];

        // Create quadrant grid
        EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);

        // Iterate over grid and check for refinement threshold
        for (auto i = 0; i < nx; i++) {
            double x = grid(XDIM, i);
            for (auto j = 0; j < ny; j++) {
                double y = grid(YDIM, j);
                double f = pde.f(x,y);
                if (fabs(f) > threshold) {
                    return 1;
                }
            }
        }

        return 0;
    },
    NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);

    // Create leaf level root patch
    double xLower = -1;
    double xUpper = 1;
    double yLower = -1;
    double yUpper = 1;
    EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKPatch rootPatch(grid);
    rootPatch.level = 0;
    rootPatch.isLeaf = true;

    // Create patch solver
    EllipticForest::FISHPACK::FISHPACKFVSolver solver{};

    // Create node factory
    EllipticForest::FISHPACK::FISHPACKPatchNodeFactory nodeFactory{};

    // Create mesh from p4est and patch solver
    EllipticForest::Mesh<EllipticForest::FISHPACK::FISHPACKPatch> mesh{MPI_COMM_WORLD, p4est, rootPatch, nodeFactory};

    // Create and run HPS method
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::FISHPACK::FISHPACKFVGrid,
        EllipticForest::FISHPACK::FISHPACKFVSolver,
        EllipticForest::FISHPACK::FISHPACKPatch,
        double>
            HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    int nSolves = 1;
    for (auto n = 0; n < nSolves; n++) {
        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            HPS.upwardsStage([&](EllipticForest::FISHPACK::FISHPACKPatch& leafPatch){
                EllipticForest::FISHPACK::FISHPACKFVGrid& grid = leafPatch.grid();
                leafPatch.vectorF() = EllipticForest::Vector<double>(grid.nPointsX() * grid.nPointsY());
                for (auto i = 0; i < grid.nPointsX(); i++) {
                    double x = grid(0, i);
                    for (auto j = 0; j < grid.nPointsY(); j++) {
                        double y = grid(1, j);
                        int index = j + i*grid.nPointsY();
                        leafPatch.vectorF()[index] = pde.f(x, y);
                    }
                }
                return;
            });
        }

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            switch (side) {
                case 0:
                    // West : Neumann
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 1:
                    // East : Neumann
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 2:
                    // South : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 3:
                    // North : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);
                
                default:
                    break;
            }
        });
    }

    // Compute error of solution
    double l1_error = 0;
    double l2_error = 0;
    double lI_error = 0;
    int nLeafPatches = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
        if (patch.isLeaf) {
            EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid();
            for (auto i = 0; i < grid.nPointsX(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.nPointsY(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.nPointsY();
                    int index_T = i + j*grid.nPointsY();
                    double diff = patch.vectorU()[index_T] - pde.u(x, y);
                    l1_error += (grid.dx()*grid.dy())*fabs(diff);
                    l2_error += (grid.dx()*grid.dy())*pow(fabs(diff), 2);
                    lI_error = fmax(lI_error, fabs(diff));
                }
            }
            nLeafPatches++;
        }
    });
    double area = (xUpper - xLower) * (yUpper - yLower);
    l1_error = l1_error / area;
    l2_error = sqrt(l2_error / area);
    int resolution = pow(2,maxLevel)*nx;
    int nDOFs = nLeafPatches * (nx * ny);

    // Compute size of quadtree and data
    double size_MB = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
        size_MB += patch.dataSize();
    });

    // Store and return results
    ResultsData results;
    results.mode = mode;
    results.min_level = minLevel;
    results.max_level = maxLevel;
    results.nx = nx;
    results.ny = ny;
    results.effective_resolution = resolution;
    results.nDOFs = nDOFs;
    results.l1_error = l1_error;
    results.l2_error = l2_error;
    results.lI_error = lI_error;
    results.build_time = app.timers["build-stage"].time();
    results.upwards_time = app.timers["upwards-stage"].time();
    results.solve_time = app.timers["solve-stage"].time();
    results.size_MB = size_MB;

    return results;

}

int main(int argc, char** argv) {

    // Initialize app
    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Set options
    app.options.setOption("cache-operators", false);
    app.options.setOption("homogeneous-rhs", false);
    app.options.setOption("refinement-threshold", 1.0);

    // Create PDE to solve
    int nPolar = 3;
    std::vector<double> x0s = {-0.5, 0.5, 0};
    std::vector<double> y0s = {-0.5, -0.5, 0.5};
    std::vector<double> r0s = {0.2, 0.3, 0.4};
    std::vector<double> r1s = {0.3, 0.4, 0.5};
    std::vector<double> ns = {3, 4, 5};
    double epsilon = 0.01;
    PoissonProblem2 pde(nPolar, x0s, y0s, r0s, r1s, ns, epsilon);

    // Set up convergence sweet parameters
    std::vector<int> patchSizeVector = {8, 16, 32, 64, 128};
    std::vector<int> levelVector = {0, 1, 2, 3, 4, 5};

    // Create storage for plotting
    std::vector<PlotPair> uniformErrorPlots;
    std::vector<PlotPair> uniformBuildTimingPlots;
    std::vector<PlotPair> uniformSolveTimingPlots;
    std::vector<PlotPair> adaptiveErrorPlots;
    std::vector<PlotPair> adaptiveBuildTimingPlots;
    std::vector<PlotPair> adaptiveSolveTimingPlots;

    // Vector of results
    std::vector<ResultsData> resultsVector;

    // Run uniform parameter sweep
    int maxResolution = pow(128, 2) * pow(2, 2*5); // About 16M DOFs
    for (auto& M : patchSizeVector) {

        PlotPair errorPair;
        PlotPair buildPair;
        PlotPair solvePair;

        for (auto& l : levelVector) {

            app.log("UNIFORM: M = %i, l = %i", M, l);
            int DOFs = pow(M, 2) * pow(2, 2*l);
            if (DOFs >= maxResolution) {
                app.log("Skipping...");
                continue;
            }

            // Set options
            app.options.setOption("min-level", l);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            ResultsData results = run(pde);
            int nDOFs = results.effective_resolution;
            double error = results.lI_error;
            resultsVector.push_back(results);

            // Save info to plot
            errorPair.first.push_back(nDOFs);
            errorPair.second.push_back(error);

            buildPair.first.push_back(nDOFs);
            buildPair.second.push_back(app.timers["build-stage"].time());

            solvePair.first.push_back(nDOFs);
            solvePair.second.push_back(app.timers["solve-stage"].time());

            // Restart timers
            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();
        }

        uniformErrorPlots.push_back(errorPair);
        uniformBuildTimingPlots.push_back(buildPair);
        uniformSolveTimingPlots.push_back(solvePair);
    }

    // Run adaptive parameter sweep
    for (auto& M : patchSizeVector) {

        PlotPair errorPair;
        PlotPair buildPair;
        PlotPair solvePair;

        for (auto& l : levelVector) {

            app.log("ADAPTIVE: M = %i, l = %i", M, l);
            int DOFs = pow(M, 2) * pow(2, 2*l);
            if (DOFs >= maxResolution) {
                app.log("Skipping...");
                continue;
            }

            // Set options
            app.options.setOption("min-level", 0);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            ResultsData results = run(pde);
            int nDOFs = results.effective_resolution;
            double error = results.lI_error;
            resultsVector.push_back(results);

            // Save info to plot
            errorPair.first.push_back(nDOFs);
            errorPair.second.push_back(error);

            buildPair.first.push_back(nDOFs);
            buildPair.second.push_back(app.timers["build-stage"].time());

            solvePair.first.push_back(nDOFs);
            solvePair.second.push_back(app.timers["solve-stage"].time());

            // Restart timers
            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();
        }

        adaptiveErrorPlots.push_back(errorPair);
        adaptiveBuildTimingPlots.push_back(buildPair);
        adaptiveSolveTimingPlots.push_back(solvePair);
    }

    // Write results to console
    app.log(ResultsData::headers());
    for (auto& results : resultsVector) {
        app.log(results.str());
    }

    // Write results to file
    std::ofstream csvFile;
    csvFile.open(pde.name() + ".csv");
    csvFile << ResultsData::headers() << std::endl;
    for (auto& results : resultsVector) {
        csvFile << results.csv() << std::endl;
    }
    csvFile.close();

    // #ifdef USE_MATPLOTLIBCPP
    // // Error plot
    // int fig1 = plt::figure(1);
    // int counter = 0;
    // std::vector<std::string> colors = {"r", "g", "b", "y", "c", "m"};
    // for (auto& [nDOFs, error] : uniformErrorPlots) {
    //     plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "--s" + colors[counter]);
    //     counter++;
    // }
    // counter = 0;
    // for (auto& [nDOFs, error] : adaptiveErrorPlots) {
    //     plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "-o" + colors[counter]);
    //     counter++;
    // }
    // std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    // std::vector<std::string> xTickLabels;
    // for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
    // plt::xlabel("Effective Resolution");
    // plt::ylabel("Inf-Norm Error");
    // plt::xticks(xTicks, xTickLabels);
    // plt::legend({{"loc", "upper right"}});
    // plt::grid(true);
    // plt::save("plot_poisson_error_" + pde.name() + "_no_title.pdf");
    // plt::title("Convergence Study - Uniform vs. Adaptive Mesh");
    // plt::save("plot_poisson_error_" + pde.name() + ".pdf");
    // plt::show();

    // int fig2 = plt::figure(2);
    // counter = 0;
    // for (auto& [nDOFs, build] : uniformBuildTimingPlots) {
    //     plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "--s" + colors[counter]);
    //     counter++;
    // }
    // counter = 0;
    // for (auto& [nDOFs, build] : adaptiveBuildTimingPlots) {
    //     plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "-o" + colors[counter]);
    //     counter++;
    // }
    // plt::xlabel("Effective Resolution");
    // plt::ylabel("Time [sec]");
    // plt::xticks(xTicks, xTickLabels);
    // plt::legend({{"loc", "lower right"}});
    // plt::grid(true);
    // plt::save("plot_poisson_build_time_" + pde.name() + "_no_title.pdf");
    // plt::title("Timing Study - Uniform vs. Adaptive Mesh - Build Stage");
    // plt::save("plot_poisson_build_time_" + pde.name() + ".pdf");
    // plt::show();

    // int fig3 = plt::figure(3);
    // counter = 0;
    // for (auto& [nDOFs, solve] : uniformSolveTimingPlots) {
    //     plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "--s" + colors[counter]);
    //     counter++;
    // }
    // counter = 0;
    // for (auto& [nDOFs, solve] : adaptiveSolveTimingPlots) {
    //     plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "-o" + colors[counter]);
    //     counter++;
    // }
    // plt::xlabel("Effective Resolution");
    // plt::ylabel("Time [sec]");
    // plt::xticks(xTicks, xTickLabels);
    // plt::legend({{"loc", "lower right"}});
    // plt::grid(true);
    // plt::save("plot_poisson_solve_time_" + pde.name() + "_no_title.pdf");
    // plt::title("Timing Study - Uniform vs. Adaptive Mesh - Solve Stage");
    // plt::save("plot_poisson_solve_time_" + pde.name() + ".pdf");
    // plt::show();
    // #endif

    return EXIT_SUCCESS;

}