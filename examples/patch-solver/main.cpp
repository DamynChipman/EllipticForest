/**
 * @file main.cpp : patch-solver
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Solves Poisson's equation on a single patch to test convergense of a single patch solver
 * 
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

/**
 * @brief Exact solution of Poisson's equation
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double uExact(double x, double y) {
    return sin(x) + sin(y);
}

/**
 * @brief RHS function to Poisson's equation
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double fRHS(double x, double y) {
    return -uExact(x, y);
}

/**
 * @brief Main driver for patch-solver
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

    // ====================================================
    // Set up convergence parameters
    // ====================================================
    std::vector<int> ns = {8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<double> errors;
    EllipticForest::Vector<double> u_exact, u_petsc;
    int nx, ny;

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver;
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FivePointStencil;
    solver.alpha_function = [&](double x, double y){
        return 1.0;
    };
    solver.beta_function = [&](double x, double y){
        return 1.0;
    };
    solver.lambda_function = [&](double x, double y){
        return 0.0;
    };

    // ====================================================
    // Run convergence sweep
    // ====================================================
    for (auto n : ns) {

        // ====================================================
        // Create patch grid
        // ====================================================
        nx = n;
        ny = n;
        double x_lower = -1;
        double x_upper = 1;
        double y_lower = -1;
        double y_upper = 1;
        EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);

        // ====================================================
        // Create boundary data
        // ====================================================
        EllipticForest::Vector<double> g_west(nx);
        EllipticForest::Vector<double> g_east(nx);
        EllipticForest::Vector<double> g_south(nx);
        EllipticForest::Vector<double> g_north(nx);
        for (int j = 0; j < ny; j++) {
            double y = grid(1, j);
            g_west[j] = uExact(x_lower, y);
            g_east[j] = uExact(x_upper, y);
        }
        for (int i = 0; i < nx; i++) {
            double x = grid(0, i);
            g_south[i] = uExact(x, y_lower);
            g_north[i] = uExact(x, y_upper);
        }
        EllipticForest::Vector<double> g = EllipticForest::concatenate({g_west, g_east, g_south, g_north});

        // ====================================================
        // Create RHS load data and exact solution
        // ====================================================
        EllipticForest::Vector<double> f(nx*ny);
        u_exact = EllipticForest::Vector<double>(nx*ny);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = grid(0, i);
                double y = grid(1, j);
                // int I = i + j*nx;
                int I = j + i*ny;
                f[I] = fRHS(x, y);
                u_exact[I] = uExact(x, y);
            }
        }

        // ====================================================
        // Solve with patch solver
        // ====================================================
        u_petsc = solver.solve(grid, g, f);

        // ====================================================
        // Compute error
        // ====================================================
        double error = EllipticForest::vectorInfNorm(u_exact, u_petsc);
        errors.push_back(error);
        app.log("N = %4i, Error = %.8e", n, error);

    }

    // ====================================================
    // Compute convergence order (should be 2nd order)
    // ====================================================
    double e1 = errors[errors.size()-2];
    double n1 = (double) ns[ns.size()-2];
    double e2 = errors[errors.size()-1];
    double n2 = (double) ns[ns.size()-1];
    double order = log(e2 / e1) / log(n1 / n2);
    app.log("Convergence Order = %.4f", order);

#ifdef USE_MATPLOTLIBCPP
    // ====================================================
    // Plot resolution vs. error
    // ====================================================
    std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<std::string> xTickLabels;
    for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));

    PyObject* py_obj;
    plt::loglog(ns, errors, "-or");
    plt::xlabel("Resolution");
    plt::xticks(xTicks, xTickLabels);
    plt::ylabel("Inf-Norm Error");
    plt::grid(true);
    plt::title("Patch Solver Convergence");
    plt::save("plot_patch_solver_convergence.pdf");
    plt::show();
#endif

    return EXIT_SUCCESS;
}