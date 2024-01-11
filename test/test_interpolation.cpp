#include "gtest/gtest.h"
#include <EllipticForest.hpp>
#include <Interpolation.hpp>

using namespace EllipticForest;

double f2d(double x, double y) {
    return x*x + y*y + 4.0;
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

    Vector<double> x1_fine = linspace(a + dx/4, b - dx/4, m*2);
    Vector<double> x2_fine = linspace(c + dy/4, d - dy/4, n*2);
    Vector<double> y_fine = I(x1_fine, x2_fine);

    Vector<double> y_fine_exp(4*m*n);
    for (int i = 0; i < 2*m; i++) {
        for (int j = 0; j < 2*n; j++) {
            y_fine_exp[j + 2*i*n] = f2d(x1_fine[i], x2_fine[j]);
        }
    }

    Vector<double> diff = y_fine_exp - y_fine;
    double max_diff = vectorInfNorm(y_fine_exp, y_fine);
    EXPECT_LT(max_diff, 1e-2);

}