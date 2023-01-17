#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include <EllipticForestApp.hpp>
#include <P4est.hpp>
#include <FISHPACK.hpp>

#if MATPLOTLIBCPP_ENABLED
namespace plt = matplotlibcpp;
#endif

using PlotPair = std::pair<std::vector<int>, std::vector<double>>;

struct ResultsData {

    std::string mode = "uniform";
    int min_level = 0;
    int max_level = 0;
    int nx = 0;
    int ny = 0;
    int effective_resolution = 0;
    int nDOFs = 0;
    double lI_error = 0;
    double l1_error = 0;
    double l2_error = 0;
    double build_time = 0;
    double upwards_time = 0;
    double solve_time = 0;

};

class GaussianPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    double x0 = 0;
    double y0 = 0;
    double sigma_x = 1;
    double sigma_y = 1;

    GaussianPoissonProblem() {}
    
    GaussianPoissonProblem(double x0, double y0, double sigma_x, double sigma_y) :
        x0(x0),
        y0(y0),
        sigma_x(sigma_x),
        sigma_y(sigma_y)
            {}

    std::string name() override { return "gaussian"; }

    double u(double x, double y) override {
        return exp(-(sigma_x*pow(x - x0, 2) + sigma_y*pow(y - y0, 2)));
    }

    double f(double x, double y) override {
        double s = u(x,y);
        return s*(-2.0*sigma_x - 2.0*sigma_y + 4.0*pow((x - x0)*sigma_x, 2) + 4.0*pow((y - y0)*sigma_y, 2));
    }

    double dudx(double x, double y) override {
        return 0;
    }

    double dudy(double x, double y) override {
        return 0;
    }

};

class PolarStarPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    int nPolar;
    std::vector<double> x0s;
    std::vector<double> y0s;
    std::vector<double> r0s;
    std::vector<double> r1s;
    std::vector<double> ns;
    double eps_disk = 0.015625;

    PolarStarPoissonProblem() {
        nPolar = 1;
        x0s = {0};
        y0s = {0};
        r0s = {0.4};
        r1s = {0.45};
        ns = {4};
    }

    PolarStarPoissonProblem(int nPolar, std::vector<double> x0s, std::vector<double> y0s, std::vector<double> r0s, std::vector<double> r1s, std::vector<double> ns) :
        nPolar(nPolar),
        x0s(x0s),
        y0s(y0s),
        r0s(r0s),
        r1s(r1s),
        ns(ns)
            {}

    std::string name() override { return "polar_star"; }

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

std::pair<int, double> solvePoissonViaHPS(EllipticForest::FISHPACK::FISHPACKProblem& pde, bool vtkFlag) {

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

    // Save initial mesh
    if (vtkFlag) {
        std::string mode;
        if (minLevel == maxLevel) mode = "uniform";
        else mode = "adaptive";
        std::string VTKFilename = "poisson_mesh_" + mode + "_" + pde.name();
        p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());
    }

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

    // Create and run HPS method
    EllipticForest::FISHPACK::FISHPACKHPSMethod HPS(pde, leafPatch, p4est);
    HPS.run();

    // Output mesh and solution
    if (vtkFlag) {
        std::string mode;
        if (minLevel == maxLevel) mode = "uniform";
        else mode = "adaptive";
        HPS.toVTK("poisson_" + mode + "_" + pde.name());
    }

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

    return {resolution, maxError};

}

int main(int argc, char** argv) {

    // Initialize app
    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Set options
    app.options.setOption("cache-operators", false);
    app.options.setOption("homogeneous-rhs", false);

    // Create PDE to solve
    // app.options.setOption("refinement-threshold", 0.1);
    // PolarStarPoissonProblem pde(
    //     2,              // Number of polar stars
    //     {-0.5, 0.5},    // x0
    //     {-0.5, 0.5},    // y0
    //     {0.1, 0.2},     // r0
    //     {0.1, 0.2},     // r1
    //     {4, 7}          // n
    // );
    app.options.setOption("refinement-threshold", 1.0);
    GaussianPoissonProblem pde(
        0.2,            // x0
        0.2,            // y0
        10,              // sigma_x
        40               // sigma_y
    );

    // Convergence parameters
    std::vector<int> patchSizeVector = {8, 16, 32, 64, 128};
    std::vector<int> levelVector {0, 1, 2, 3, 4};
    // std::vector<int> patchSizeVector = {4, 8, 32};
    // std::vector<int> levelVector {0, 1, 3};

    // Create storage for plotting
    std::vector<PlotPair> uniformErrorPlots;
    std::vector<PlotPair> uniformBuildTimingPlots;
    std::vector<PlotPair> uniformSolveTimingPlots;
    std::vector<PlotPair> adaptiveErrorPlots;
    std::vector<PlotPair> adaptiveBuildTimingPlots;
    std::vector<PlotPair> adaptiveSolveTimingPlots;

    // Run uniform parameter sweep
    bool vtkFlag = false;
    for (auto& M : patchSizeVector) {

        PlotPair errorPair;
        PlotPair buildPair;
        PlotPair solvePair;

        for (auto& l : levelVector) {

            // Set options
            app.options.setOption("min-level", l);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            if (M == 128 && l == 4) vtkFlag = true;
            else vtkFlag = false;
            auto [nDOFs, error] = solvePoissonViaHPS(pde, vtkFlag);

            // Output to console
            app.log("M = %i", M);
            app.log("l = %i", l);
            app.log("nDOFs = %i", nDOFs);
            app.log("error = %24.16e", error);
            app.log("build-time = %f sec", app.timers["build-stage"].time());
            app.log("upwards-time = %f sec", app.timers["upwards-stage"].time());
            app.log("solve-time = %f sec", app.timers["solve-stage"].time());

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

            // Set options
            app.options.setOption("min-level", 0);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            if (M == 128 && l == 4) vtkFlag = true;
            else vtkFlag = false;
            auto [nDOFs, error] = solvePoissonViaHPS(pde, vtkFlag);

            // Output to console
            app.log("M = %i", M);
            app.log("l = %i", l);
            app.log("nDOFs = %i", nDOFs);
            app.log("error = %24.16e", error);
            app.log("build-time = %f sec", app.timers["build-stage"].time());
            app.log("upwards-time = %f sec", app.timers["upwards-stage"].time());
            app.log("solve-time = %f sec", app.timers["solve-stage"].time());

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
    // for (auto& M : patchSizeVector) {
    //     for (auto& l : levelVector) {

    //         // Set options
    //         app.options.setOption("min-level", 0);
    //         app.options.setOption("max-level", l);
    //         app.options.setOption("nx", M);
    //         app.options.setOption("ny", M);

    //         // Solve via HPS
    //         if (M == 128 && l == 4) vtkFlag = true;
    //         else vtkFlag = false;
    //         auto [nDOFs, error] = solvePoissonViaHPS(pde, vtkFlag);

    //         // Output to console
    //         app.log("M = %i", M);
    //         app.log("l = %i", l);
    //         app.log("nDOFs = %i", nDOFs);
    //         app.log("error = %24.16e", error);
    //         app.log("build-time = %f sec", app.timers["build-stage"].time());
    //         app.log("upwards-time = %f sec", app.timers["upwards-stage"].time());
    //         app.log("solve-time = %f sec", app.timers["solve-stage"].time());

    //         // Restart timers
    //         app.timers["build-stage"].restart();
    //         app.timers["upwards-stage"].restart();
    //         app.timers["solve-stage"].restart();
    //     }
    // }

    #if MATPLOTLIBCPP_ENABLED
    // Error plot
    int fig1 = plt::figure(1);
    int counter = 0;
    std::vector<std::string> colors = {"r", "g", "b", "y", "c", "m"};
    for (auto& [nDOFs, error] : uniformErrorPlots) {
        plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "--s" + colors[counter]);
        counter++;
    }
    counter = 0;
    for (auto& [nDOFs, error] : adaptiveErrorPlots) {
        plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "-o" + colors[counter]);
        counter++;
    }
    std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<std::string> xTickLabels;
    for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
    plt::xlabel("Effective Resolution");
    plt::ylabel("Inf-Norm Error");
    // plt::title("Convergence Study - Uniform vs. Adaptive Mesh");
    plt::xticks(xTicks, xTickLabels);
    plt::legend({{"loc", "upper right"}});
    plt::grid(true);
    plt::save("plot_poisson_error_" + pde.name() + "_no_title.pdf");
    // plt::show();

    int fig2 = plt::figure(2);
    counter = 0;
    for (auto& [nDOFs, build] : uniformBuildTimingPlots) {
        plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "--s" + colors[counter]);
        counter++;
    }
    counter = 0;
    for (auto& [nDOFs, build] : adaptiveBuildTimingPlots) {
        plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "-o" + colors[counter]);
        counter++;
    }
    plt::xlabel("Effective Resolution");
    plt::ylabel("Time [sec]");
    // plt::title("Timing Study - Uniform vs. Adaptive Mesh - Build Stage");
    plt::xticks(xTicks, xTickLabels);
    plt::legend({{"loc", "lower right"}});
    plt::grid(true);
    plt::save("plot_poisson_build_time_" + pde.name() + "_no_title.pdf");
    // plt::show();

    int fig3 = plt::figure(3);
    counter = 0;
    for (auto& [nDOFs, solve] : uniformSolveTimingPlots) {
        plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "--s" + colors[counter]);
        counter++;
    }
    counter = 0;
    for (auto& [nDOFs, solve] : adaptiveSolveTimingPlots) {
        plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "-o" + colors[counter]);
        counter++;
    }
    plt::xlabel("Effective Resolution");
    plt::ylabel("Time [sec]");
    // plt::title("Timing Study - Uniform vs. Adaptive Mesh - Solve Stage");
    plt::xticks(xTicks, xTickLabels);
    plt::legend({{"loc", "lower right"}});
    plt::grid(true);
    plt::save("plot_poisson_solve_time_" + pde.name() + "_no_title.pdf");
    // plt::show();
    #endif

    return EXIT_SUCCESS;
}