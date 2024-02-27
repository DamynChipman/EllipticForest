#include "gtest/gtest.h"
#include <EllipticForest.hpp>
#include <Interpolation.hpp>

using namespace EllipticForest;

double f1d(double x) {
    return sin(x) + 4.0;
}

double f2d(double x, double y) {
    return 30000.0*(sin(x) + pow(sin(y), 3));
    // return x*x + y*y + 4.0;
    // return x + y;
}

TEST(Interpolation, linear) {

    int n = 6;
    Vector<double> xx = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    Vector<double> yy(n);
    for (int i = 0; i < yy.size(); i++) {
        yy[i] = xx[i]*xx[i];
    }

    LinearInterpolant f(xx, yy);

    double x = 0.5;
    double y_test = f(x);
    double y_true = x*x;

    EXPECT_NEAR(y_test, y_true, 1e-1);

    Vector<double> xs = {0.1, 0.3, 0.5, 0.7, 0.9};
    Vector<double> ys_test = f(xs);
    Vector<double> ys_true(5);
    for (int i = 0; i < ys_true.size(); i++) {
        ys_true[i] = xs[i]*xs[i];
    }

    for (int i = 0; i < ys_true.size(); i++) {
        EXPECT_NEAR(ys_test[i], ys_true[i], 1e-1);
    }

}

TEST(Interpolation, polynomial) {
    
    int n = 16;
    Vector<double> x = linspace<double>(-2, 2, n);
    Vector<double> y(n);
    for (int i = 0; i < y.size(); i++) {
        y[i] = f1d(x[i]);
    }

    PolynomialInterpolant f(x, y, 2);

    double x_test = 0.5;
    double y_test = f(x_test);
    double y_true = f1d(x_test);
    double error = fabs(y_test - y_true);

    EXPECT_NEAR(y_test, y_true, 1e-2);

    Vector<double> xv = linspace<double>(-2, 0, n);
    Vector<double> yv_test = f(xv);
    Vector<double> yv_true(n);
    for (int i = 0; i < n; i++) {
        yv_true[i] = f1d(xv[i]);
        // printf("x = %f, y_test = %f, y_true = %f, error = %e\n", xv[i], yv_test[i], yv_true[i], fabs(yv_test[i] - yv_true[i]));
    }

    for (int i = 0; i < yv_true.size(); i++) {
        EXPECT_NEAR(yv_test[i], yv_true[i], 1e-2);
    }

}

TEST(Interpolation, bilinear) {

    double a = -1;
    double b = 1;
    double c = 0;
    double d = 2;
    int m = 16;
    int n = 16;
    double dx = (b - a) / m;
    double dy = (d - c) / n;
    Vector<double> x1_coarse = linspace(a + dx/2, b - dx/2, m);
    Vector<double> x2_coarse = linspace(c + dy/2, d - dy/2, n);
    Vector<double> y_coarse(m*n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            y_coarse[j + i*n] = f2d(x1_coarse[i], x2_coarse[j]);
        }
    }

    BilinearInterpolant I(x1_coarse, x2_coarse, y_coarse);

    b = b / 2.;
    d = d / 2.;
    Vector<double> x1_fine = linspace(a + dx/4, b - dx/4, m*2);
    Vector<double> x2_fine = linspace(c + dy/4, d - dy/4, n*2);
    Vector<double> y_test = I(x1_fine, x2_fine);

    Vector<double> y_true(4*m*n);
    Vector<double> rel_err(4*m*n);
    for (int i = 0; i < 2*m; i++) {
        for (int j = 0; j < 2*n; j++) {
            int ii = j + 2*i*n;
            y_true[ii] = f2d(x1_fine[i], x2_fine[j]);
            rel_err[ii] = fabs(y_test[ii] - y_true[ii]) / y_true[ii];
            // printf("x1 = %f, x2 = %f, y_test = %f, y_true = %f, error = %e\n", x1_fine[i], x2_fine[j], y_test[ii], y_true[ii], rel_err[ii]);
        }
    }

    double max_error = *std::max(rel_err.data().begin(), rel_err.data().end());
    EXPECT_LT(max_error, 1e-2);

}

// TEST(Interpolation, polynomial_2d) {

//     double a = 0;
//     double b = 1;
//     double c = 0;
//     double d = 1;
//     int m = 4;
//     int n = 4;
//     double dx = (b - a) / m;
//     double dy = (d - c) / n;
//     Vector<double> x1_coarse = linspace(a + dx/2, b - dx/2, m);
//     Vector<double> x2_coarse = linspace(c + dy/2, d - dy/2, n);
//     Vector<double> y_coarse(m*n);
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             y_coarse[j + i*n] = f2d(x1_coarse[i], x2_coarse[j]);
//         }
//     }

//     Polynomial2DInterpolant I(x1_coarse, x2_coarse, y_coarse, 1, 1);

//     Vector<double> x1_fine = linspace(a + dx/4, b - dx/4, m*2);
//     Vector<double> x2_fine = linspace(c + dy/4, d - dy/4, n*2);
//     Vector<double> y_test = I(x1_fine, x2_fine);

//     Vector<double> y_true(4*m*n);
//     for (int i = 0; i < 2*m; i++) {
//         for (int j = 0; j < 2*n; j++) {
//             y_true[j + 2*i*n] = f2d(x1_fine[i], x2_fine[j]);
//             printf("x1 = %f, x2 = %f, y_test = %f, y_true = %f, error = %e\n", x1_fine[i], x2_fine[j], y_test[j + 2*i*n], f2d(x1_fine[i], x2_fine[j]), fabs(f2d(x1_fine[i], x2_fine[j]) - y_test[j + 2*i*n]));
//         }
//     }

//     Vector<double> diff = y_true - y_test;
//     double max_diff = vectorInfNorm(y_true, y_test);
//     EXPECT_LT(max_diff, 1e-2);

// }

// TEST(Interpolation, polynomial_2d) {

//     double a = -1;
//     double b = 1;
//     double c = 0;
//     double d = 2;
//     int m = 16;
//     int n = 16;
//     double dx = (b - a) / m;
//     double dy = (d - c) / n;
//     Vector<double> x1_coarse = linspace(a + dx/2, b - dx/2, m);
//     Vector<double> x2_coarse = linspace(c + dy/2, d - dy/2, n);
//     Vector<double> y_coarse(m*n);
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             y_coarse[j + i*n] = f2d(x1_coarse[i], x2_coarse[j]);
//         }
//     }

//     Polynomial2DInterpolant I(x1_coarse, x2_coarse, y_coarse, 2, 2);

//     Vector<double> x1_fine = linspace(a + dx/4, b - dx/4, m*2);
//     Vector<double> x2_fine = linspace(c + dy/4, d - dy/4, n*2);
//     Vector<double> y_test = I(x1_fine, x2_fine);

//     Vector<double> y_true(4*m*n);
//     for (int i = 0; i < 2*m; i++) {
//         for (int j = 0; j < 2*n; j++) {
//             y_true[j + 2*i*n] = f2d(x1_fine[i], x2_fine[j]);
//             printf("x1 = %f, x2 = %f, y_test = %f, y_true = %f, error = %e\n", x1_fine[i], x2_fine[j], y_test[j + 2*i*n], f2d(x1_fine[i], x2_fine[j]), fabs(f2d(x1_fine[i], x2_fine[j]) - y_test[j + 2*i*n]));
//         }
//     }

//     Vector<double> diff = y_true - y_test;
//     double max_diff = vectorInfNorm(y_true, y_test);
//     EXPECT_LT(max_diff, 1e-2);

// }