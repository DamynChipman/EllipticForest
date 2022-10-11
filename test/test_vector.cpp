#include "gtest/gtest.h"
#include <EllipticForestApp.hpp>
#include <Vector.hpp>

using namespace EllipticForest;

TEST(Vector, init) {

    Vector<double> v1{};
    EXPECT_EQ(v1.size(), 0);

    Vector<double> v2(4);
    EXPECT_EQ(v2.size(), 4);
    EXPECT_EQ(v2.data().size(), 4);

    double* d3 = (double*) malloc(4*sizeof(double));
    d3[0] = 0; d3[1] = 10; d3[2] = 20; d3[3] = 30;
    Vector<double> v3(4, d3);
    EXPECT_EQ(v3.size(), 4);
    EXPECT_EQ(v3.data().size(), 4);
    // EXPECT_EQ(v3.data().data(), d3);
    for (auto i = 0; i < 4; i++) {
        EXPECT_EQ(v3.data()[i], d3[i]);
    }
    
    Vector<double> v4(4, 3.14);
    EXPECT_EQ(v4.size(), 4);
    for (auto i = 0; i < 4; i++) {
        EXPECT_EQ(v4[i], 3.14);
    }

    Vector<double> v5({0, 1, 2, 3});
    EXPECT_EQ(v5.size(), 4);
    EXPECT_EQ(v5.data().size(), 4);
    for (auto i = 0; i < 4; i++) {
        EXPECT_EQ(v5(i), (double) i);
    }

}

TEST(Vector, data) {

    

}