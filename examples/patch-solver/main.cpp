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
    // Sine/Cosine
    // return sin(x) + sin(y);

    // Variable coefficient
    // return x*(1. - x)*y*(1. - y)*exp(x*y);
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
    // Sine/Cosine
    // return -uExact(x, y);

    // Variable coefficient
    // return M_PI*(exp(x*y)*(1 - x)*x*(1 - y) - exp(x*y)*(1 - x)*x*y + exp(x*y)*(1 - x)*exp(2)*(1 - y)*y)*cos(M_PI*x)*cos(M_PI*y) - M_PI*(exp(x*y)*(1 - x)*(1 - y)*y - exp(x*y)*x*(1 - y)*y + exp(x*y)*(1 - x)*x*(1 - y)*exp(2))*sin(M_PI*x)*sin(M_PI*y) + (-2*exp(x*y)*(1 - x)*x + 2*exp(x*y)*(1 - x)*exp(2)*(1 - y) - 2*exp(x*y)*(1 - x)*exp(2)*y + exp(x*y)*(1 - x)*exp(3)*(1 - y)*y)*(2 + cos(M_PI*x)*sin(M_PI*y)) + (-2*exp(x*y)*(1 - y)*y + 2*exp(x*y)*(1 - x)*(1 - y)*exp(2) - 2*exp(x*y)*x*(1 - y)*exp(2) + exp(x*y)*(1 - x)*x*(1 - y)*exp(3))*(2 + cos(M_PI*x)*sin(M_PI*y));
    return pow(cos(y),2)*sin(x) + pow(cos(x),2)*sin(y) - sin(x)*(2 + sin(x)*sin(y)) - sin(y)*(2 + sin(x)*sin(y));
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
        // return 1.0;
        // return cos(M_PI*x)*sin(M_PI*y) + 2.;
        return sin(x)*sin(y) + 2.;
    };
    solver.lambda_function = [&](double x, double y){
        return 0.0;
    };

    // ====================================================
    // Run convergence sweep
    // ====================================================
    app.addTimer("solve-time");
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
        auto solve_timer = app.timers["solve-time"];
        solve_timer.restart();
        solve_timer.start();
        u_petsc = solver.solve(grid, g, f);
        solve_timer.stop();
        double dt_solve = solve_timer.time();

        // ====================================================
        // Compute error
        // ====================================================
        double error = EllipticForest::vectorInfNorm(u_exact, u_petsc);
        errors.push_back(error);
        app.logHead("N = %4i, Error = %16.8e, Time = %16.8f [sec]", n, error, dt_solve);

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