/**
 * @file main.cpp : poisson
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Sets up and solves Poisson's equation using the Hierarchical Poincaré-Steklov method
 * 
 * Solves Poisson's equation:
 * 
 * laplacian( u ) = f
 * 
 * subject to Dirichlet boundary conditions provided by the exact solution.
 * 
 * By default, this is set to solve for the exact solution:
 * 
 * u(x,y) = sin(x) + sin(y)
 * 
 * thus,
 * 
 * f(x,y) = -sin(x) - sin(y) = -u(x,y).
 * 
 * In this example, we run a convergence analysis by solving via the HPS method with varying
 * levels of refinement and patch size. If configured with `matplotlibcpp`, then the error, build,
 * and solve time plots will be shown.
 * 
 */

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>
#include <fstream>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

using PlotPair = std::pair<std::vector<int>, std::vector<double>>;

/**
 * @brief User-defined solution function
 * 
 * When computing error, calls this function to compare with numerical solution.
 * 
 * Used by EllipticForest to create the boundary data.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double uFunction(double x, double y) {
    return sin(x) + sin(y);
}

/**
 * @brief User-defined load function
 * 
 * Right-hand side function of the elliptic PDE.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double fFunction(double x, double y) {
    return -uFunction(x, y);
}

/**
 * @brief User-defined alpha function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double alphaFunction(double x, double y) {
    return 1.0;
}

/**
 * @brief User-defined beta function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double betaFunction(double x, double y) {
    return 0.0;
}

/**
 * @brief User-defined lambda function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double lambdaFunction(double x, double y) {
    return 0.0;
}

/**
 * @brief Convert a number to a string using a format specifier
 * 
 * @param x The number
 * @param format Format specifier for number output
 * @return std::string The number as a string
 */
std::string number2string(double x, std::string format="%.4f") {
    char buffer[32];
    snprintf(buffer, 32, format.c_str(), x);
    return std::string(buffer);
}

/**
 * @brief A data structure to hold the results of the program
 */
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
    double size_MB = 0;

    std::string csv() {
        std::string res = "";
        res += mode + ",";
        res += std::to_string(min_level) + ",";
        res += std::to_string(max_level) + ",";
        res += std::to_string(nx) + ",";
        res += std::to_string(ny) + ",";
        res += std::to_string(effective_resolution) + ",";
        res += std::to_string(nDOFs) + ",";
        res += number2string(lI_error, "%.16e") + ",";
        res += number2string(l1_error, "%.16e") + ",";
        res += number2string(l2_error, "%.16e") + ",";
        res += number2string(build_time, "%.16e") + ",";
        res += number2string(upwards_time, "%.16e") + ",";
        res += number2string(solve_time, "%.16e") + ",";
        res += number2string(size_MB, "%.16e") + ",";
        return res;
    }

    std::string str() {
        std::string res = "";
        res += mode + "  ";
        res += std::to_string(min_level) + "  ";
        res += std::to_string(max_level) + "  ";
        res += std::to_string(nx) + "  ";
        res += std::to_string(ny) + "  ";
        res += std::to_string(effective_resolution) + "  ";
        res += std::to_string(nDOFs) + "  ";
        res += std::to_string(lI_error) + "  ";
        res += std::to_string(l1_error) + "  ";
        res += std::to_string(l2_error) + "  ";
        res += std::to_string(build_time) + "  ";
        res += std::to_string(upwards_time) + "  ";
        res += std::to_string(solve_time) + "  ";
        res += std::to_string(size_MB) + "  ";
        return res;
    }

    static std::string headers() {
        std::string res = "";
        res += "mode,";
        res += "min_level,";
        res += "max_level,";
        res += "nx,";
        res += "ny,";
        res += "effective_resolution,";
        res += "nDOFs,";
        res += "lI_error,";
        res += "l1_error,";
        res += "l2_error,";
        res += "build_time,";
        res += "upwards_time,";
        res += "solve_time,";
        res += "size_MB,";
        return res;
    }

};

/**
 * @brief Does a single solve of the Poisson problem via HPS method
 * 
 * @param vtk_flag Flag to output VTK files
 * @return ResultsData Data structure with results of solver
 */
ResultsData solveEllipticViaHPS() {

    // ====================================================
    // Get the options
    // ====================================================
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    int n_solves = std::get<int>(app.options["n-solves"]);
    int min_level = std::get<int>(app.options["min-level"]);
    int max_level = std::get<int>(app.options["max-level"]);
    std::string mode = min_level == max_level ? "uniform" : "adaptive";
    int nx = std::get<int>(app.options["nx"]);
    int ny = std::get<int>(app.options["ny"]);
    double threshold = std::get<double>(app.options["refinement-threshold"]);
    double x_lower = std::get<double>(app.options["x-lower"]);
    double x_upper = std::get<double>(app.options["x-upper"]);
    double y_lower = std::get<double>(app.options["y-lower"]);
    double y_upper = std::get<double>(app.options["y-upper"]);
    bool vtk_flag = std::get<bool>(app.options["vtk-flag"]);

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // ====================================================
    // Create node factory and mesh
    // ====================================================
    EllipticForest::FiniteVolumeNodeFactory nodeFactory{};
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            double f = fFunction(x, y);
            return fabs(f) > threshold;
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        nodeFactory
    );

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver{};
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

    // ====================================================
    // Create and run HPS solver
    // ====================================================
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::FiniteVolumeGrid,
         EllipticForest::FiniteVolumeSolver,
         EllipticForest::FiniteVolumePatch,
         double>
            HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    for (auto n = 0; n < n_solves; n++) {

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });

    }

    // ====================================================
    // Write solution and functions to file
    // ====================================================
    if (vtk_flag) {
        
        // Extract out solution and right-hand side data stored on leaves
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        
        EllipticForest::Vector<double> fMesh{};
        fMesh.name() = "f_rhs";
        
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
                fMesh.append(patch.vectorF());
            }
            return 1;
        });

        // Add mesh functions to mesh
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction(fMesh);
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.alpha_function(x, y);
            },
            "alpha_fn"
        );
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.beta_function(x, y);
            },
            "beta_fn"
        );
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.lambda_function(x, y);
            },
            "lambda_fn"
        );
        mesh.addMeshFunction(
            [&](double x, double y){
                return uFunction(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic");

    }

    // ====================================================
    // Compute error and timing results
    // ====================================================
    double l1_error = 0;
    double l2_error = 0;
    double lI_error = 0;
    int n_leaf_patches = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();
            for (auto i = 0; i < grid.nx(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.ny(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.ny();
                    double diff = patch.vectorU()[index] - uFunction(x, y);
                    l1_error += (grid.dx()*grid.dy())*fabs(diff);
                    l2_error += (grid.dx()*grid.dy())*pow(fabs(diff), 2);
                    lI_error = fmax(lI_error, fabs(diff));
                }
            }
            n_leaf_patches++;
        }
        return 1;
    });
    double area = (x_upper - x_lower) * (y_upper - y_lower);
    l1_error = l1_error / area;
    l2_error = sqrt(l2_error / area);
    int resolution = pow(2,max_level)*nx;
    int nDOFs = n_leaf_patches * (nx * ny);

    // Compute size of quadtree and data
    double size_MB = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::FiniteVolumePatch& patch){
        size_MB += patch.dataSize();
    });

    // Store and return results
    ResultsData results;
    results.mode = mode;
    results.min_level = min_level;
    results.max_level = max_level;
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

    // ====================================================
    // Initialize app and MPI
    // ====================================================
    EllipticForest::EllipticForestApp app(&argc, &argv);
    EllipticForest::MPI::MPIObject mpi(MPI_COMM_WORLD);

    // ====================================================
    // Setup options
    // ====================================================
    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);
    
    double threshold = 1.2;
    app.options.setOption("refinement-threshold", threshold);
    
    int n_solves = 1;
    app.options.setOption("n-solves", n_solves);

    double x_lower = 0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = M_PI;
    app.options.setOption("x-upper", x_upper);

    double y_lower = 0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = M_PI;
    app.options.setOption("y-upper", y_upper);

    bool vtk_flag = false;
    app.options.setOption("vtk-flag", vtk_flag);

    // ====================================================
    // Setup convergence analysis parameters
    // ====================================================
    std::vector<int> patch_size_vector = {4, 8, 16, 32};
    std::vector<int> min_level_vector = {0, 0, 0, 0};
    std::vector<int> max_level_vector = {1, 2, 3, 4, 5, 6, 7};

    // Create storage for plotting
    std::vector<PlotPair> uniform_error_plots;
    std::vector<PlotPair> uniform_build_timing_plots;
    std::vector<PlotPair> uniform_solve_timing_plots;
    std::vector<PlotPair> adaptive_error_plots;
    std::vector<PlotPair> adaptive_build_timing_plots;
    std::vector<PlotPair> adaptive_solve_timing_plots;

    // Vector of results
    std::vector<ResultsData> results_vector;

    // ====================================================
    // Run uniform convergence analysis
    // ====================================================
    int max_resolution = pow(2, 20);
    for (auto& M : patch_size_vector) {

        PlotPair error_pair;
        PlotPair build_pair;
        PlotPair solve_pair;

        for (auto& l : max_level_vector) {

            int DOFs = pow(M, 2) * pow(2, 2*l);
            app.logHead("UNIFORM: M = %i, l = %i, DOFs = %i", M, l, DOFs);
            if (DOFs > max_resolution) {
                app.log("Skipping...");
                continue;
            }

            // Set options
            app.options.setOption("min-level", l);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            ResultsData results = solveEllipticViaHPS();
            int nDOFs = results.effective_resolution;
            double error = results.l1_error;
            results_vector.push_back(results);

            // Save info to plot
            error_pair.first.push_back(nDOFs);
            error_pair.second.push_back(error);

            build_pair.first.push_back(nDOFs);
            build_pair.second.push_back(app.timers["build-stage"].time());

            solve_pair.first.push_back(nDOFs);
            solve_pair.second.push_back(app.timers["solve-stage"].time());

            // Restart timers
            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();
        }

        uniform_error_plots.push_back(error_pair);
        uniform_build_timing_plots.push_back(build_pair);
        uniform_solve_timing_plots.push_back(solve_pair);
    }

    // ====================================================
    // Run adaptive convergence analysis
    // ====================================================
    for (auto& M : patch_size_vector) {

        PlotPair error_pair;
        PlotPair build_pair;
        PlotPair solve_pair;

        for (auto& l : max_level_vector) {

            int DOFs = pow(M, 2) * pow(2, 2*l);
            app.logHead("ADAPTIVE: M = %i, l = %i, DOFs = %i", M, l, DOFs);
            if (DOFs > max_resolution) {
                app.log("Skipping...");
                continue;
            }

            // Set options
            app.options.setOption("min-level", 0);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            ResultsData results = solveEllipticViaHPS();
            int nDOFs = results.effective_resolution;
            double error = results.l1_error;
            results_vector.push_back(results);

            // Save info to plot
            error_pair.first.push_back(nDOFs);
            error_pair.second.push_back(error);

            build_pair.first.push_back(nDOFs);
            build_pair.second.push_back(app.timers["build-stage"].time());

            solve_pair.first.push_back(nDOFs);
            solve_pair.second.push_back(app.timers["solve-stage"].time());

            // Restart timers
            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();
        }

        adaptive_error_plots.push_back(error_pair);
        adaptive_build_timing_plots.push_back(build_pair);
        adaptive_solve_timing_plots.push_back(solve_pair);
    }

    // ====================================================
    // Output results
    // ====================================================
    // Write to console
    app.logHead(ResultsData::headers());
    for (auto& results : results_vector) {
        app.logHead(results.str());
    }

    // Write results to file
    std::ofstream csv_file;
    std::string results_filename = "elliptic_results_" + std::to_string(mpi.getRank()) + ".csv";
    csv_file.open(results_filename.c_str());
    csv_file << ResultsData::headers() << std::endl;
    for (auto& results : results_vector) {
        csv_file << results.csv() << std::endl;
    }
    csv_file.close();
    
    // ====================================================
    // Make and output plots
    // ====================================================
    #ifdef USE_MATPLOTLIBCPP
    if (mpi.getRank() == EllipticForest::MPI::HEAD_RANK) {

        // Error plot
        int fig1 = plt::figure(1);
        int counter = 0;
        std::vector<std::string> colors = {"r", "g", "b", "y", "c", "m"};
        for (auto& [nDOFs, error] : uniform_error_plots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patch_size_vector[counter]), nDOFs, error, "-v" + colors[counter]);
            counter++;
        }
        counter = 0;
        for (auto& [nDOFs, error] : adaptive_error_plots) {
            plt::named_loglog("Adaptive: N = " + std::to_string(patch_size_vector[counter]), nDOFs, error, "--o" + colors[counter]);
            counter++;
        }
        std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
        std::vector<std::string> xTickLabels;
        for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
        plt::xlabel("Effective Resolution");
        plt::ylabel("Inf-Norm Error");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "upper right"}});
        plt::grid(true);
        plt::save("plot_elliptic_error_no_title.pdf");
        plt::title("Convergence Study - Uniform vs. Adaptive Mesh");
        plt::save("plot_elliptic_error.pdf");
        plt::show();

        // Build time plot
        int fig2 = plt::figure(2);
        counter = 0;
        for (auto& [nDOFs, build] : uniform_build_timing_plots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patch_size_vector[counter]), nDOFs, build, "--s" + colors[counter]);
            counter++;
        }
        counter = 0;
        for (auto& [nDOFs, build] : adaptive_build_timing_plots) {
            plt::named_loglog("Adaptive: N = " + std::to_string(patch_size_vector[counter]), nDOFs, build, "-o" + colors[counter]);
            counter++;
        }
        plt::xlabel("Effective Resolution");
        plt::ylabel("Time [sec]");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "lower right"}});
        plt::grid(true);
        plt::save("plot_elliptic_build_time_no_title.pdf");
        plt::title("Timing Study - Uniform vs. Adaptive Mesh - Build Stage");
        plt::save("plot_elliptic_build_time.pdf");
        plt::show();

        // Solve time plot
        int fig3 = plt::figure(3);
        counter = 0;
        for (auto& [nDOFs, solve] : uniform_solve_timing_plots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patch_size_vector[counter]), nDOFs, solve, "--s" + colors[counter]);
            counter++;
        }
        counter = 0;
        for (auto& [nDOFs, solve] : adaptive_solve_timing_plots) {
            plt::named_loglog("Adaptive: N = " + std::to_string(patch_size_vector[counter]), nDOFs, solve, "-o" + colors[counter]);
            counter++;
        }
        plt::xlabel("Effective Resolution");
        plt::ylabel("Time [sec]");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "lower right"}});
        plt::grid(true);
        plt::save("plot_elliptic_solve_time_no_title.pdf");
        plt::title("Timing Study - Uniform vs. Adaptive Mesh - Solve Stage");
        plt::save("plot_elliptic_solve_time.pdf");
        plt::show();

    }
    #endif

    return EXIT_SUCCESS;
}