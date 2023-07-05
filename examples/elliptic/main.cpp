#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <FISHPACK.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

double besselJ(int n, double x) {
    return jn(n, x);
}

const double kappa = 8.0;

double uExact(double x, double y) {
    // Laplace 1
    // return sin(2.0*M_PI*x) * sinh(2.0*M_PI*y);

    // Poisson 1
    return sin(2.0*M_PI*x) + cos(2.0*M_PI*y);

    // Helmholtz 1
    // double x0 = -2;
    // double y0 = 0;
    // return besselJ(0, kappa*sqrt(pow(x - x0, 2) + pow(y - y0, 2)));

    // Variable Poisson 1 (just set BC)
    // return 4.0;
}

double fRHS(double x, double y) {
    // Laplace 1
    // return 0.0;

    // Poisson 1
    return -4.0*pow(M_PI, 2)*uExact(x, y);

    // Helmholtz 1
    // return 0.0;

    // Variable Poisson 1
    // if (-0.5 < x && x < 1 && -0.5 < y && y < 1) {
    //     return 10.;
    // }
    // else {
    //     return 0.;
    // }
}

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);

    std::vector<int> ns = {16, 32, 64, 128, 256, 512};
    std::vector<double> errors;
    EllipticForest::Vector<double> u_exact, u_petsc;
    int nx, ny;

    EllipticForest::Petsc::PetscPatchSolver solver;
    solver.setAlphaFunction([&](double x, double y){
        return 1.0;
    });
    solver.setBetaFunction([&](double x, double y){
        return 1.0;
    });
    solver.setLambdaFunction([&](double x, double y){
        return pow(kappa,2);
    });
    // EllipticForest::FISHPACK::FISHPACKFVSolver solver;

    for (auto n : ns) {

        nx = n;
        ny = n;
        double x_lower = -1;
        double x_upper = 1;
        double y_lower = -1;
        double y_upper = 1;
        EllipticForest::Petsc::PetscGrid grid(nx, ny, x_lower, x_upper, y_lower, y_upper);
        // EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, x_lower, x_upper, y_lower, y_upper);

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

        u_petsc = solver.solve(grid, g, f);

        double error = EllipticForest::vectorInfNorm(u_exact, u_petsc);
        errors.push_back(error);
        app.log("N = %4i, Error = %.8e", n, error);

    }

    EllipticForest::Vector<float> u_petsc_float(u_petsc.size());
    for (int i = 0; i < u_petsc.size(); i++) {
        u_petsc_float[i] = static_cast<float>(u_petsc[i]);
    }

    PyObject* py_obj;
    plt::imshow(u_petsc_float.dataPointer(), nx, ny, 1, {}, &py_obj);
    plt::colorbar(py_obj);
    plt::title("u_petsc");
    plt::show();

    return EXIT_SUCCESS;
}