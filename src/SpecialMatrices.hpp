#ifndef SPECIAL_MATRICES_HPP_
#define SPECIAL_MATRICES_HPP_

#include "Matrix.hpp"

namespace EllipticForest {

template<typename NumericalType>
class IdentityMatrix : public Matrix<NumericalType> {

public:

    /**
     * @brief Construct a new Identity Matrix object of size `N`
     * 
     * @param N Size of matrix
     */
    IdentityMatrix(std::size_t N) :
        Matrix<NumericalType>(N, N, 0) {

        for (auto i = 0; i < N; i++) {
            this->operator()(i,i) = 1;
        }

    }

};

template<typename NumericalType>
class ParallelIdentityMatrix : public ParallelMatrix<NumericalType> {

public:

    ParallelIdentityMatrix(std::size_t N) :
        ParallelMatrix<NumericalType>(MPI_COMM_WORLD) {

        //
        MatCreateConstantDiagonal(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, 1.0, &this->mat);
        MatGetLocalSize(this->mat, &this->local_rows, &this->local_cols);
        MatGetSize(this->mat, &this->global_rows, &this->global_cols);
        this->is_created = true;

    }

    ParallelIdentityMatrix(MPI::Communicator comm, std::size_t N) :
        ParallelMatrix<NumericalType>(comm) {

        //
        MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, N, N, 1.0, &this->mat);
        MatSetFromOptions(this->mat);
        MatGetLocalSize(this->mat, &this->local_rows, &this->local_cols);
        MatGetSize(this->mat, &this->global_rows, &this->global_cols);
        this->is_created = true;

    }

    ParallelIdentityMatrix(MPI::Communicator comm, std::size_t N, Petsc::MatType matrix_type) :
        ParallelMatrix<NumericalType>(comm) {

        //
        MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, N, N, 1.0, &this->mat);
        MatSetType(this->mat, matrix_type);
        MatGetLocalSize(this->mat, &this->local_rows, &this->local_cols);
        MatGetSize(this->mat, &this->global_rows, &this->global_cols);
        this->is_created = true;

    }

};

template<typename NumericalType>
class DiagonalMatrix : public Matrix<NumericalType> {

public:

    /**
     * @brief Construct a new Diagonal Matrix object with entries from `diag`
     * 
     * @param diag Vector of diagonal entries
     */
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

    /**
     * @brief Construct a new Interpolation Matrix Fine 2 Coarse object
     * 
     * @param ncoarse Number of coarse cells
     */
    InterpolationMatrixFine2Coarse(std::size_t ncoarse) :
        Matrix<NumericalType>(ncoarse, 2*ncoarse, 0) {

        for (auto i = 0; i < this->nrows_; i++) {
            for (auto j = 0; j < this->ncols_; j++) {
                if (j == 2*i) {
                    this->operator()(i,j) = 0.5;
                    this->operator()(i,j+1) = 0.5;
                }
            }
        }

    }

};

// template<typename NumericalType>
// class ParallelInterpolationMatrixFine2Coarse : public ParallelMatrix<NumericalType> {

// public:

//     ParallelInterpolationMatrixFine2Coarse(std::size_t n_coarse) :
//         ParallelMatrix<NumericalType>(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n_coarse, 2*n_coarse, MATAIJ) {

//         //
//         Vector<int> is_row = vectorRange(0, n_coarse-1);
//         Vector<int> is_cols = vectorRange(0, )

//     }

// };


template<typename NumericalType>
class InterpolationMatrixCoarse2Fine : public Matrix<NumericalType> {

public:

    /**
     * @brief Construct a new Interpolation Matrix Coarse 2 Fine object
     * 
     * @param nfine Number of fine cells
     */
    InterpolationMatrixCoarse2Fine(std::size_t nfine) :
        Matrix<NumericalType>(nfine, nfine/2, 0) {

        int k = 3;
        Vector<NumericalType> edge_coefficients = {1.40625, -0.5625, 0.15625};
        for (auto i = 0; i < this->nrows_; i++) {
            for (auto j = 0; j < this->ncols_; j++) {
                if (i == 0 && j == 0) {
                    this->operator()(i,j) = edge_coefficients[0];
                    this->operator()(i,j+1) = edge_coefficients[1];
                    this->operator()(i,j+2) = edge_coefficients[2];
                }
                else if (i == this->nrows_-1 && j == this->ncols_-k) {
                    this->operator()(i,j) = edge_coefficients[2];
                    this->operator()(i,j+1) = edge_coefficients[1];
                    this->operator()(i,j+2) = edge_coefficients[0];
                }
                else if (i == 2*j+1 && i%2 == 1 && i != this->nrows_-1) {
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