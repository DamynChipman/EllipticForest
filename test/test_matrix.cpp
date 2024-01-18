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

TEST(Matrix, index_sets) {

    Matrix<double> m1(6, 8, {
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 5, 6, 7, 8, 9,
        3, 4, 5, 6, 7, 8, 9, 10,
        4, 5, 6, 7, 8, 9, 10, 11,
        5, 6, 7, 8, 9, 10, 11, 12,
        6, 7, 8, 9, 10, 11, 12, 13
    });

    Vector<int> I1 = {1, 3, 5};
    Vector<int> J1 = {2, 3, 4, 5};

    Matrix<double> m2_expected(3, 4, {
        4, 5, 6, 7,
        6, 7, 8, 9,
        8, 9, 10, 11
    });

    Matrix<double> m2_test = m1.getFromIndexSet(I1, J1);

    EXPECT_EQ(m2_test.nRows(), m2_expected.nRows());
    EXPECT_EQ(m2_test.nCols(), m2_expected.nCols());
    for (auto i = 0; i < m2_test.nRows(); i++) {
        for (auto j = 0; j < m2_test.nCols(); j++) {
            EXPECT_EQ(m2_test(i,j), m2_expected(i,j));
        }
    }

}

TEST(Matrix, block_diagonal) {

    Matrix<double> m1(2, 2, {
        1, 1,
        1, 1
    });
    Matrix<double> m2(2, 2, {
        2, 2,
        2, 2
    });
    Matrix<double> m3(2, 2, {
        3, 3,
        3, 3
    });
    std::vector<Matrix<double>> diag = {m1, m2, m3};

    Matrix<double> m2_test = blockDiagonalMatrix(diag);
    Matrix<double> m2_expected(6, 6, {
        1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0,
        0, 0, 2, 2, 0, 0,
        0, 0, 2, 2, 0, 0,
        0, 0, 0, 0, 3, 3,
        0, 0, 0, 0, 3, 3
    });

    EXPECT_EQ(m2_test.nRows(), m2_expected.nRows());
    EXPECT_EQ(m2_test.nCols(), m2_expected.nCols());
    for (auto i = 0; i < m2_test.nRows(); i++) {
        for (auto j = 0; j < m2_test.nCols(); j++) {
            EXPECT_EQ(m2_test(i,j), m2_expected(i,j));
        }
    }

}

TEST(Matrix, set_block) {

    Matrix<double> m1_test(4, 4, {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4
    });
    Matrix<double> m2(2, 2, {
        100, 100,
        100, 100
    });

    m1_test.setBlock(1, 1, m2);

    Matrix<double> m1_expected(4, 4, {
        1, 1, 1, 1,
        2, 100, 100, 2,
        3, 100, 100, 3,
        4, 4, 4, 4
    });

    EXPECT_EQ(m1_test.nRows(), m1_expected.nRows());
    EXPECT_EQ(m1_test.nCols(), m1_expected.nCols());
    for (auto i = 0; i < m1_test.nRows(); i++) {
        for (auto j = 0; j < m1_test.nCols(); j++) {
            EXPECT_EQ(m1_test(i,j), m1_expected(i,j));
        }
    }

}

TEST(Matrix, linear_solve) {

    Matrix<double> A(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 10
    });
    Vector<double> b = {1, 1, 1};

    Vector<double> x_test = solve(A, b);
    Vector<double> x_expected = {-1, 1, 0};

    for (auto i = 0; i < x_test.size(); i++) {
        EXPECT_NEAR(x_test[i], x_expected[i], 1e-15);
    }

    Matrix<double> B(3, 4, {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4
    });

    Matrix<double> X_test = solve(A, B);
    Matrix<double> X_expected(3, 4, {
        -1, -2, -3, -4,
        1, 2, 3, 4,
        0, 0, 0, 0
    });

    EXPECT_EQ(X_test.nRows(), X_expected.nRows());
    EXPECT_EQ(X_test.nCols(), X_expected.nCols());
    for (auto i = 0; i < X_test.nRows(); i++) {
        for (auto j = 0; j < X_test.nCols(); j++) {
            EXPECT_NEAR(X_test(i,j), X_expected(i,j), 1e-14);
        }
    }

}

TEST(Matrix, math) {

    Matrix<double> m1(3, 4, {
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6
    });
    Matrix<double> m2(3, 4, {
        4, 5, 6, 7,
        5, 6, 7, 8,
        6, 7, 8, 9
    });

    Matrix<double> m3_test = m2 + m1;
    Matrix<double> m4_test = m2 - m1;
    Matrix<double> m5_test = 2.0*m1;

    Matrix<double> m3_expected(3, 4, {
        5, 7, 9, 11,
        7, 9, 11, 13,
        9, 11, 13, 15
    });
    Matrix<double> m4_expected(3, 4, {
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3
    });
    Matrix<double> m5_expected(3, 4, {
        2, 4, 6, 8,
        4, 6, 8, 10,
        6, 8, 10, 12
    });

    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 4; j++) {
            EXPECT_EQ(m3_test(i,j), m3_expected(i,j));
            EXPECT_EQ(m4_test(i,j), m4_expected(i,j));
            EXPECT_EQ(m5_test(i,j), m5_expected(i,j));
        }
    }

    Matrix<double> m6(2, 3, {
        1, 2, 3,
        4, 5, 6
    });
    Matrix<double> m7(3, 2, {
        1, 2,
        2, 1,
        2, 3
    });

    Matrix<double> m8_test = m6 * m7;
    Matrix<double> m8_expected(2, 2, {
        11, 13,
        26, 31
    });

    EXPECT_EQ(m8_test.nRows(), m8_expected.nRows());
    EXPECT_EQ(m8_test.nCols(), m8_expected.nCols());
    for (auto i = 0; i < m8_test.nRows(); i++) {
        for (auto j = 0; j < m8_test.nCols(); j++) {
            EXPECT_EQ(m8_test(i,j), m8_expected(i,j));
        }
    }

    Vector<double> v1 = {2, 2, 2};
    Vector<double> v2_test = m6 * v1;
    Vector<double> v2_expected = {12, 30};

    EXPECT_EQ(v2_test.size(), v2_expected.size());
    for (auto i = 0; i < v2_test.size(); i++) {
        EXPECT_EQ(v2_test[i], v2_expected[i]);
    }

}

TEST(Matrix, block_permute) {

    Matrix<double> m1(4, 6, {
        1, 1, 2, 2, 3, 3,
        1, 1, 2, 2, 3, 3,
        4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9
    });

    Vector<int> pi_rows = {2, 1, 0};
    Vector<int> pi_cols = {0, 2, 1};
    Vector<int> R = {2, 1, 1};
    Vector<int> C = {2, 2, 2};

    Matrix<double> m2_expected(4, 6, {
        7, 7, 9, 9, 8, 8,
        4, 4, 6, 6, 5, 5,
        1, 1, 3, 3, 2, 2,
        1, 1, 3, 3, 2, 2
    });

    Matrix<double> m2_test = m1.blockPermute(pi_rows, pi_cols, R, C);

    EXPECT_EQ(m2_test.nRows(), m2_expected.nRows());
    EXPECT_EQ(m2_test.nCols(), m2_expected.nCols());
    for (auto i = 0; i < m2_test.nRows(); i++) {
        for (auto j = 0; j < m2_test.nCols(); j++) {
            EXPECT_EQ(m2_test(i,j), m2_expected(i,j));
        }
    }

}