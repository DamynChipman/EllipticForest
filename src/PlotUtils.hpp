#ifndef PLOT_UTILS_HPP_
#define PLOT_UTILS_HPP_

#if USE_MATPLOTLIBCPP

#include <matplotlibcpp.h>
#include "Vector.hpp"
#include "Matrix.hpp"

namespace matplotlibcpp {

/**
 * @brief Utility to plot matrix entires
 * 
 * @tparam NumericalType 
 * @param A Matrix to plot
 * @param tolerance Tolerance for a zero
 */
template<typename NumericalType>
void matshow(EllipticForest::Matrix<NumericalType>& A, double tolerance = 1e-12) {

    EllipticForest::Vector<int> x = EllipticForest::vectorRange(0, A.nRows()-1);
    EllipticForest::Vector<int> y = EllipticForest::vectorRange(0, A.nCols()-1);

    figure_size(512, 512);

    std::vector<double> row, col, colors;
    for (auto j = 0; j < A.nCols(); j++) {
        for (auto i = 0; i < A.nRows(); i++) {
            if (fabs(A(i,j)) > tolerance) {
                row.push_back((double) i);
                col.push_back((double) A.nRows() - j);
                colors.push_back((double) A(i,j));
            }
        }
    }
    scatter(row, col, 1, {{"marker", "s"}, {"color", "k"}});

}

} // NAMESPACE : matplotlibcpp

#endif

#endif // PLOT_UTILS_HPP_