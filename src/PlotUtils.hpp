#ifndef PLOT_UTILS_HPP_
#define PLOT_UTILS_HPP_

#if USE_MATPLOTLIBCPP

#include <utility>
#include <matplotlibcpp.h>
#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"

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

template<typename NumericalType>
std::pair<EllipticForest::Vector<NumericalType>, EllipticForest::Vector<NumericalType>> meshgrid(EllipticForest::Vector<NumericalType>& x1, EllipticForest::Vector<NumericalType>& x2) {

    auto M = x1.size();
    auto N = x2.size();
    EllipticForest::Vector<NumericalType> x1_mesh(M*N);
    EllipticForest::Vector<NumericalType> x2_mesh(M*N);
    for (auto i = 0; i < M; i++) {
        for (auto j = 0; j < N; j++) {
            auto ii = j + i*N;
            x1_mesh[ii] = x1[i];
            x2_mesh[ii] = x2[j];
        }
    }
    return {x1_mesh, x2_mesh};

}

template<typename NumericalType>
bool scatter3(EllipticForest::PatchGridBase<NumericalType>& grid, EllipticForest::Vector<NumericalType>& values) {

    auto x1_points = grid.xPoints();
    auto x2_points = grid.yPoints();
    auto [x1_mesh, x2_mesh] = meshgrid(x1_points, x2_points);
    scatter(x1_mesh.data(), x2_mesh.data(), values.data());

}

} // NAMESPACE : matplotlibcpp

#endif

#endif // PLOT_UTILS_HPP_