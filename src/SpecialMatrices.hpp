#ifndef SPECIAL_MATRICES_HPP_
#define SPECIAL_MATRICES_HPP_

#include "Matrix.hpp"

namespace EllipticForest {

template<typename NumericalType>
class IdentityMatrix : public Matrix<NumericalType> {

public:

    IdentityMatrix(std::size_t N) :
        Matrix<NumericalType>(N, N, 0) {

        for (auto i = 0; i < N; i++) {
            this->operator()(i,i) = 1;
        }

    }

};

template<typename NumericalType>
class DiagonalMatrix : public Matrix<NumericalType> {

public:

    DiagonalMatrix(Vector<NumericalType> diag) :
        Matrix<NumericalType>(diag.size(), diag.size(), 0) {

        for (auto i = 0; i < diag.size(); i++) {
            this->operator()(i, i) = diag[i];
        }

    }

};

template<typename NumericalType>
class InterpolationMatrixFine2Coarse : public Matrix<NumericalType> {

public:

    InterpolationMatrixFine2Coarse(std::size_t nCoarse) :
        Matrix<NumericalType>(nCoarse, 2*nCoarse, 0) {

        for (auto i = 0; i < this->nRows_; i++) {
            for (auto j = 0; j < this->nCols_; j++) {
                if (j == 2*i) {
                    this->operator()(i,j) = 0.5;
                    this->operator()(i,j+1) = 0.5;
                }
            }
        }

    }

};


template<typename NumericalType>
class InterpolationMatrixCoarse2Fine : public Matrix<NumericalType> {

public:

    InterpolationMatrixCoarse2Fine(std::size_t nFine) :
        Matrix<NumericalType>(nFine, nFine/2, 0) {

        int k = 3;
        Vector<NumericalType> edgeCoefs = {1.40625, -0.5625, 0.15625};
        for (auto i = 0; i < this->nRows_; i++) {
            for (auto j = 0; j < this->nCols_; j++) {
                if (i == 0 && j == 0) {
                    this->operator()(i,j) = edgeCoefs[0];
                    this->operator()(i,j+1) = edgeCoefs[1];
                    this->operator()(i,j+2) = edgeCoefs[2];
                }
                else if (i == this->nRows_-1 && j == this->nCols_-k) {
                    this->operator()(i,j) = edgeCoefs[2];
                    this->operator()(i,j+1) = edgeCoefs[1];
                    this->operator()(i,j+2) = edgeCoefs[0];
                }
                else if (i == 2*j+1 && i%2 == 1 && i != this->nRows_-1) {
                    this->operator()(i,j) = 0.75;
                    this->operator()(i,j+1) = 0.25;
                    this->operator()(i+1,j) = 0.25;
                    this->operator()(i+1,j+1) = 0.75;
                }
            }
        }

    }

};

} // NAMESPACE : EllipticForest

#endif // SPECIAL_MATRICES_HPP_