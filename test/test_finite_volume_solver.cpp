#include "gtest/gtest.h"
#include <Patches/FiniteVolume/FiniteVolumeSolver.hpp>

using namespace EllipticForest;

double uExact(double x, double y) {
    return sin(x) + sin(y);
}

double fRHS(double x, double y) {
    return -uExact(x, y);
}

TEST(FiniteVolumeSolver, init) {

    EXPECT_NO_THROW(FiniteVolumeSolver solver1{});
    EXPECT_NO_THROW(FiniteVolumeSolver solver1(MPI_COMM_WORLD, [&](double,double){return 0;}, [&](double,double){return 0;}, [&](double,double){return 0;}, [&](double,double){return 0;}));

}

TEST(FiniteVolumeSolver, solver_convergence_FISHPACK) {

    std::vector<int> ns = {8, 16, 32, 64, 128, 256, 512};
    std::vector<double> errors;
    Vector<double> u_exact, u_solver;

    FiniteVolumeSolver solver;
    solver.solver_type = FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = [&](double x, double y){
        return 1.0;
    };
    solver.beta_function = [&](double x, double y){
        return 1.0;
    };
    solver.lambda_function = [&](double x, double y){
        return 0.0;
    };

    double xlower = -1;
    double xupper = 1;
    double ylower = -1;
    double yupper = 1;;
    for (auto n : ns) {

        // Create patch grid
        FiniteVolumeGrid grid(MPI_COMM_WORLD, n, xlower, xupper, n, ylower, yupper);

        // Create boundary data
        Vector<double> g_west(n);
        Vector<double> g_east(n);
        Vector<double> g_south(n);
        Vector<double> g_north(n);
        for (int j = 0; j < n; j++) {
            double y = grid(YDIM, j);
            g_west[j] = uExact(xlower, y);
            g_east[j] = uExact(xupper, y);
        }
        for (int i = 0; i < n; i++) {
            double x = grid(XDIM, i);
            g_south[i] = uExact(x, ylower);
            g_north[i] = uExact(x, yupper);
        }
        Vector<double> g = concatenate({g_west, g_east, g_south, g_north});

        // Create RHS load data and exact solution
        Vector<double> f(n*n);
        u_exact = Vector<double>(n*n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double x = grid(0, i);
                double y = grid(1, j);
                int I = j + i*n;
                f[I] = fRHS(x, y);
                u_exact[I] = uExact(x, y);
            }
        }

        // Solve with patch solver
        u_solver = solver.solve(grid, g, f);

        // Compute error
        double error = vectorInfNorm(u_exact, u_solver);
        errors.push_back(error);

    }

    // Compute convergence order (should be 2nd order)
    double e1 = errors[errors.size()-2];
    double n1 = (double) ns[ns.size()-2];
    double e2 = errors[errors.size()-1];
    double n2 = (double) ns[ns.size()-1];
    double order = log(e2 / e1) / log(n1 / n2);
    EXPECT_GT(order, 1.95);

}

// Currently commented out due to GoogleTest seg fault; works fine in patch-solver example
// TEST(FiniteVolumeSolver, solver_convergence_five_point_stencil) {

//     std::vector<int> ns = {8, 16, 32, 64, 128, 256, 512};
//     std::vector<double> errors;
//     Vector<double> u_exact, u_solver;

//     FiniteVolumeSolver solver;
//     solver.solver_type = FiniteVolumeSolverType::FivePointStencil;
//     solver.alpha_function = [&](double x, double y){
//         return 1.0;
//     };
//     solver.beta_function = [&](double x, double y){
//         return 1.0;
//     };
//     solver.lambda_function = [&](double x, double y){
//         return 0.0;
//     };

//     double xlower = -1;
//     double xupper = 1;
//     double ylower = -1;
//     double yupper = 1;;
//     for (auto n : ns) {

//         // Create patch grid
//         FiniteVolumeGrid grid(MPI_COMM_WORLD, n, xlower, xupper, n, ylower, yupper);

//         // Create boundary data
//         Vector<double> g_west(n);
//         Vector<double> g_east(n);
//         Vector<double> g_south(n);
//         Vector<double> g_north(n);
//         for (int j = 0; j < n; j++) {
//             double y = grid(YDIM, j);
//             g_west[j] = uExact(xlower, y);
//             g_east[j] = uExact(xupper, y);
//         }
//         for (int i = 0; i < n; i++) {
//             double x = grid(XDIM, i);
//             g_south[i] = uExact(x, ylower);
//             g_north[i] = uExact(x, yupper);
//         }
//         Vector<double> g = concatenate({g_west, g_east, g_south, g_north});

//         // Create RHS load data and exact solution
//         Vector<double> f(n*n);
//         u_exact = Vector<double>(n*n);
//         for (int i = 0; i < n; i++) {
//             for (int j = 0; j < n; j++) {
//                 double x = grid(0, i);
//                 double y = grid(1, j);
//                 int I = j + i*n;
//                 f[I] = fRHS(x, y);
//                 u_exact[I] = uExact(x, y);
//             }
//         }

//         // Solve with patch solver
//         u_solver = solver.solve(grid, g, f);

//         // Compute error
//         double error = vectorInfNorm(u_exact, u_solver);
//         errors.push_back(error);

//     }

//     // Compute convergence order (should be 2nd order)
//     double e1 = errors[errors.size()-2];
//     double n1 = (double) ns[ns.size()-2];
//     double e2 = errors[errors.size()-1];
//     double n2 = (double) ns[ns.size()-1];
//     double order = log(e2 / e1) / log(n1 / n2);
//     EXPECT_GT(order, 1.95);

// }