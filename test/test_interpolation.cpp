#include "gtest/gtest.h"
#include <EllipticForest.hpp>
#include <Interpolation.hpp>

using namespace EllipticForest;

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