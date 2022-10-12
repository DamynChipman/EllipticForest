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

    Vector<double> v6(v5);
    EXPECT_EQ(v6.size(), v5.size());
    EXPECT_NE(&v6, &v5);
    for (auto i = 0; i < v6.size(); i++) {
        EXPECT_EQ(v6[i], v5[i]);
    }
    v6[1] = 10;
    EXPECT_NE(v6[1], v5[1]);

    Vector<double> v7 = v4;
    EXPECT_EQ(v7.size(), v4.size());
    EXPECT_NE(&v7, &v4);
    for (auto i = 0; i < v7.size(); i++) {
        EXPECT_EQ(v7[i], v4[i]);
    }
    v7[1] = 10;
    EXPECT_NE(v7[1], v4[1]);
    

}