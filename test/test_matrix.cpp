#include "gtest/gtest.h"
#include <EllipticForestApp.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>

using namespace EllipticForest;

TEST(Matrix, transpose) {

    Matrix<double> m1(2, 3, {
        10, 20, 30,
        40, 50, 60
    });

    Matrix<double> m2(3, 2, {
        10, 40,
        20, 50,
        30, 60
    });

    Matrix<double> m1T = m1.T();
    for (auto i = 0; i < m1T.nRows(); i++) {
        for (auto j = 0; j < m1T.nCols(); j++) {
            EXPECT_EQ(m1T(i,j), m2(i,j));
        }
    }

}

TEST(Matrix, range) {

    Matrix<double> m1(3, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    });

    Matrix<double> m2(2, 3, {
        2, 3, 4,
        6, 7, 8
    });

    Matrix<double> m3 = m1(0, 1, 1, 3);

    EXPECT_EQ(m3.nRows(), m2.nRows());
    EXPECT_EQ(m3.nCols(), m2.nCols());
    for (auto i = 0; i < m3.nRows(); i++) {
        for (auto j = 0; j < m3.nCols(); j++) {
            EXPECT_EQ(m3(i,j), m2(i,j));
        }
    }

}

TEST(Matrix, get_row_and_col) {

    Matrix<double> m1(3, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    });

    Vector<double> v1({9, 10, 11, 12});

    Vector<double> v2 = m1.getRow(2);
    EXPECT_EQ(v2.size(), v1.size());

    for (auto i = 0; i < v2.size(); i++) {
        EXPECT_EQ(v2[i], v1[i]);
    }

    Vector<double> v3({2, 6, 10});

    Vector<double> v4 = m1.getCol(1);
    EXPECT_EQ(v3.size(), v3.size());

    for (auto i = 0; i < v4.size(); i++) {
        EXPECT_EQ(v3[i], v4[i]);
    }

}