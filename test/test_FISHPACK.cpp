#include "gtest/gtest.h"
#include <algorithm>
#include <EllipticForestApp.hpp>
#include <Vector.hpp>
#include <FISHPACK.hpp>

using namespace EllipticForest;
using namespace EllipticForest::FISHPACK;

TEST(FISHPACK, grid) {

    std::size_t nx = 4;
    std::size_t ny = 4;
    double xL = -1;
    double xU = 1;
    double yL = -1;
    double yU = 1;
    FISHPACKFVGrid grid(nx, ny, xL, xU, yL, yU);

    EXPECT_EQ(grid.name(), "FISHPACKFVGrid");
    EXPECT_EQ(grid.nx(), nx);
    EXPECT_EQ(grid.ny(), ny);
    EXPECT_EQ(grid.xLower(), xL);
    EXPECT_EQ(grid.xUpper(), xU);
    EXPECT_EQ(grid.yLower(), yL);
    EXPECT_EQ(grid.yUpper(), yU);
    EXPECT_EQ(grid.dx(), (xU - xL)/nx);
    EXPECT_EQ(grid.dy(), (yU - yL)/ny);

}

// TEST(FISHPACK, patch_solver) {

//     std::vector<std::size_t> resolutions = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
//     std::vector<double> errors;
//     for (auto& r : resolutions) {
//         // Create grid
//         std::size_t nx = r;
//         std::size_t ny = r;
//         double xL = -1;
//         double xU = 1;
//         double yL = -1;
//         double yU = 1;
//         FISHPACKFVGrid grid(nx, ny, xL, xU, yL, yU);

//         // Create Poisson equation
//         FISHPACKProblem pde;
//         pde.setU([](double x, double y){
//             return x*x + y*y;
//         });
//         pde.setF([](double x, double y){
//             return 4.0;
//         });
//         pde.setDUDX([](double x, double y){
//             return x;
//         });
//         pde.setDUDY([](double x, double y){
//             return y;
//         });

//         // Create Dirichlet and RHS data
//         std::size_t nBoundary = 2*grid.nx() + 2*grid.ny();
//         Vector<double> g(nBoundary);
//         Vector<int> IS_West = vectorRange(0, grid.ny() - 1);
//         Vector<int> IS_East = vectorRange(grid.ny(), 2*grid.ny() - 1);
//         Vector<int> IS_South = vectorRange(2*grid.ny(), 2*grid.ny() + grid.nx() - 1);
//         Vector<int> IS_North = vectorRange(2*grid.ny() + grid.nx(), 2*grid.ny() + 2*grid.nx() - 1);
//         Vector<int> IS_WESN = concatenate({IS_West, IS_East, IS_South, IS_North});
//         for (auto i = 0; i < nBoundary; i++) {
//             std::size_t iSide = i % grid.nx();
//             if (std::find(IS_West.data().begin(), IS_West.data().end(), i) != IS_West.data().end()) {
//                 double x = grid.xLower();
//                 double y = grid(YDIM, iSide);
//                 g[i] = pde.u(x, y);
//             }
//             if (std::find(IS_East.data().begin(), IS_East.data().end(), i) != IS_East.data().end()) {
//                 double x = grid.xUpper();
//                 double y = grid(YDIM, iSide);
//                 g[i] = pde.u(x, y);
//             }
//             if (std::find(IS_South.data().begin(), IS_South.data().end(), i) != IS_South.data().end()) {
//                 double x = grid(XDIM, iSide);
//                 double y = grid.yLower();
//                 g[i] = pde.u(x, y);
//             }
//             if (std::find(IS_North.data().begin(), IS_North.data().end(), i) != IS_North.data().end()) {
//                 double x = grid(XDIM, iSide);
//                 double y = grid.yUpper();
//                 g[i] = pde.u(x, y);
//             }
//         }

//         std::size_t nInterior = grid.nx() * grid.ny();
//         Vector<double> f(nInterior);
//         Vector<double> u_expected(nInterior);
//         for (auto i = 0; i < grid.nx(); i++) {
//             for (auto j = 0; j < grid.ny(); j++) {
//                 double x = grid(XDIM, i);
//                 double y = grid(YDIM, j);
//                 std::size_t idx = j + i*grid.ny();
//                 f[idx] = pde.f(x, y);
//                 u_expected[idx] = pde.u(x, y);
//             }
//         }

//         // Solve via FISHPACK
//         FISHPACKFVSolver solver;
//         Vector<double> u_test = solver.solve(grid, g, f);

//         // Compare solutions
//         double infNorm = vectorInfNorm(u_test, u_expected);
//         errors.push_back(infNorm);
//         std::cout << "[" << nx << ", " << ny << "] : Inf-Norm = " << infNorm << std::endl;
//     }

//     EXPECT_LT(errors[errors.size()-1], 1e-6);

// }