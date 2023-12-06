#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <string>
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "EllipticForestApp.hpp"
#include "Vector.hpp"

// ====================================================================================================
// FORTRAN wrappers for LAPACK routines
// ====================================================================================================
extern "C" {
    /**
     * @brief Double, general matrix-vector multiplication: y = alpha*A*x + beta*y
     * 
     * @param TRANS `N`, `T`, or `C` flags for transpose
     * @param M Number of rows in A
     * @param N Number of columns in A
     * @param ALPHA Mat-vec multiplier
     * @param A Double array of matrix A data
     * @param LDA Leading dimension of A
     * @param X Double array of vector x data
     * @param INCX Increment for the elements of x
     * @param BETA Vector y multiplier
     * @param Y Double array of vector y data (input and output)
     * @param INCY Increment for the elements of y
     */
    void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);

    /**
     * @brief Double, general matrix-matrix multiplication: C = alpha*A*B + beta*C
     * 
     * @param TRANSA `N`, `T`, or `C` flags for transpose of A
     * @param TRANSB `N`, `T`, or `C` flags for transpose of B
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Number of columns of A
     * @param ALPHA Mat-mat multiplier
     * @param A Double array of matrix A data
     * @param LDA Leading dimension of A
     * @param B Double array of matrix B data
     * @param LDB Leading dimension of B
     * @param BETA Matrix C multiplier
     * @param C Double array of matrix C data
     * @param LDC Leading dimension of C
     */
    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);

    /**
     * @brief Double, general linear solver for system A*X = B
     * 
     * @param N Number of rows and columns in A
     * @param NRHS Number of right-hand sides, or columns in B
     * @param A Double array of matrix A data
     * @param LDA Leading dimension of A
     * @param IPIV Pivot indices
     * @param B Double array of matrix B data; on output, stores solution of X
     * @param LDB Leading dimension of B
     * @param INFO Info flag
     */
    void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
}

namespace EllipticForest {

namespace Petsc {
    using Mat = Mat;
    using MatType = MatType;
    using MatAssemblyType = MatAssemblyType;
    using MatReuse = MatReuse;
} // NAMESPACE : Petsc

template<typename NumericalType>
class Matrix {

protected:

    /**
     * @brief Number of rows
     * 
     */
    std::size_t nrows_;

    /**
     * @brief Number of columns
     * 
     */
    std::size_t ncols_;

    /**
     * @brief Underlying storage
     * 
     */
    std::vector<NumericalType> data_;

public:

    /**
     * @brief Construct a new Matrix object (default)
     * 
     */
    Matrix() :
        nrows_(0),
        ncols_(0),
        data_(0)
            {}

    /**
     * @brief Construct a new Matrix object with space for `nrows` x `ncols` matrix
     * 
     * @param nrows Number of rows
     * @param ncols Number of columns
     */
    Matrix(std::size_t nrows, std::size_t ncols) :
        nrows_(nrows),
        ncols_(ncols),
        data_(nrows * ncols)
            {}

    /**
     * @brief Construct a new Matrix object where every entry is the value of `value`
     * 
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @param value Value of all entires
     */
    Matrix(std::size_t nrows, std::size_t ncols, NumericalType value) :
        nrows_(nrows),
        ncols_(ncols),
        data_(nrows * ncols, value)
            {}

    /**
     * @brief Construct a new Matrix object from a linear vector `data` in row-major format
     * 
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @param data Vector of data in row-major format
     */
    Matrix(std::size_t nrows, std::size_t ncols, std::vector<NumericalType> data) :
        nrows_(nrows),
        ncols_(ncols),
        data_(data)
            {}

    /**
     * @brief Construct a new Matrix object from an initializer list
     * 
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @param init_list Initialization list
     */
    Matrix(std::size_t nrows, std::size_t ncols, std::initializer_list<NumericalType> init_list) :
        nrows_(nrows),
        ncols_(ncols),
        data_(init_list)
            {}

    /**
     * @brief Construct a new Matrix object from another matrix (copy constructor)
     * 
     * @param A Other matrix
     */
    Matrix(const Matrix& A) :
        nrows_(A.nrows()),
        ncols_(A.ncols()),
        data_(A.data())
            {}

    /**
     * @brief Copy assignment operator
     * 
     * @param rhs RHS matrix
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator=(const Matrix<NumericalType>& rhs) {
        if (&rhs != this) {
            nrows_ = rhs.nrows();
            ncols_ = rhs.ncols();
            data_ = rhs.data();
            return *this;
        }
        return *this;
    }

    /**
     * @brief Returns the number of rows
     * 
     * @return std::size_t 
     */
    std::size_t nrows() const { return nrows_; }

    /**
     * @brief Returns the number of rows
     * 
     * @return std::size_t 
     */
    std::size_t nRows() const { return nrows_; }

    /**
     * @brief Returns the number of columns
     * 
     * @return std::size_t 
     */
    std::size_t ncols() const { return ncols_; }

    /**
     * @brief Returns the number of columns
     * 
     * @return std::size_t 
     */
    std::size_t nCols() const { return ncols_; }

    /**
     * @brief Returns the data vector
     * 
     * @return std::vector<NumericalType> 
     */
    std::vector<NumericalType> data() const { return data_; }

    /**
     * @brief Non-const version of the data vector
     * 
     * @return std::vector<NumericalType>& 
     */
    std::vector<NumericalType>& dataNoConst() { return data_; }

    /**
     * @brief Returns the raw data pointer
     * 
     * @return NumericalType* 
     */
    NumericalType* dataPointer() { return data_.data(); }

    /**
     * @brief Getter function for entry A[i,j]
     * 
     * @param i Row index
     * @param j Column index
     * @return NumericalType 
     */
    NumericalType getEntry(std::size_t i, std::size_t j) {
        return data_[flattenIndex_(i,j)];
    }

    /**
     * @brief Getter function for entry A[i,j]
     * 
     * @param i Row index
     * @param j Column index
     * @return NumericalType& 
     */
    NumericalType& operator()(std::size_t i, std::size_t j) {
        return data_[flattenIndex_(i,j)];
    }

    /**
     * @brief Getter function for range of entries A[a:b, c:d]
     * 
     * @param a Lower row index
     * @param b Upper row index
     * @param c Lower column index
     * @param d Upper column index
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        Matrix<NumericalType> res((b - a) + 1, (d - c) + 1);
        for (auto i = 0; i < res.nrows(); i++) {
            for (auto j = 0; j < res.ncols(); j++) {
                res(i,j) = data_[flattenIndex_(a + i, c + j)];
            }
        }
        return res;
    }

    /**
     * @brief Getter function for row A[i,:]
     * 
     * @param row_index Row index 
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> getRow(std::size_t row_index) {
        Vector<NumericalType> res(ncols_);
        Matrix<NumericalType> rowMatrix = operator()(row_index, row_index, 0, ncols_ - 1);
        for (auto j = 0; j < ncols_; j++) {
            res[j] = rowMatrix(0, j);
        }
        return res;
    }

    /**
     * @brief Getter function for column A[:,j]
     * 
     * @param col_index Column index
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> getCol(std::size_t col_index) {
        Vector<NumericalType> res(nrows_);
        Matrix<NumericalType> colMatrix = operator()(0, nrows_ - 1, col_index, col_index);
        for (auto i = 0; i < ncols_; i++) {
            res[i] = colMatrix(i, 0);
        }
        return res;
    }

    /**
     * @brief Accessor function for A[I,J] where I and J are index sets
     * 
     * @param I Row index set
     * @param J Column index set
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> getFromIndexSet(Vector<int> I, Vector<int> J) {
        if (I.size() > nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::getFromIndexSet] Size of index set `I` is greater than number of rows in `this`:\n";
            error_msg += "\tI.size() = " + std::to_string(I.size()) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (J.size() > ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::getFromIndexSet] Size of index set `J` is greater than number of columns in `this`:\n";
            error_msg += "\tJ.size() = " + std::to_string(J.size()) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }

        Matrix<NumericalType> res(I.size(), J.size());
        for (auto i = 0; i < I.size(); i++) {
            for (auto j = 0; j < J.size(); j++) {
                if (I[i] > nrows_ || I[i] < 0) {
                    std::string error_msg = "[EllipticForest::Matrix::getFromIndexSet] Index in `I` is out of range:\n";
                    error_msg += "\ti = " + std::to_string(i) + "\n";
                    error_msg += "\tI[i] = " + std::to_string(I[i]) + "\n";
                    error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
                }
                if (J[j] > ncols_ || J[j] < 0) {
                    std::string error_msg = "[EllipticForest::Matrix::getFromIndexSet] Index in `J` is out of range:\n";
                    error_msg += "\tj = " + std::to_string(j) + "\n";
                    error_msg += "\tJ[j] = " + std::to_string(J[j]) + "\n";
                    error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
                }
                res(i,j) = operator()(I[i], J[j]);
            }
        }
        return res;

    }

    /**
     * @brief Accessor function for A[I,J] where I and J are index sets
     * 
     * @param I Row index set
     * @param J Column index set
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> operator()(Vector<int> I, Vector<int> J) {
        return getFromIndexSet(I,J);
    }

    /**
     * @brief Setter function for row A[i,:]
     * 
     * @param row_index Row index
     * @param vec Vector to set
     */
    void setRow(std::size_t row_index, Vector<NumericalType>& vec) {
        if (row_index > nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::setRow] `row_index` exceeds matrix size:\n";
            error_msg += "\trowIndex = " + std::to_string(row_index) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (vec.size() != ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::setRow] Size of `vec` is not the same as number of columns in `this`:\n";
            error_msg += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        for (auto j = 0; j < ncols_; j++) {
            operator()(row_index, j) = vec[j];
        }

    }

    /**
     * @brief Setter for column A[:,j]
     * 
     * @param col_index Column index
     * @param vec Vector to set
     */
    void setColumn(std::size_t col_index, Vector<NumericalType>& vec) {
        if (col_index > ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::setColumn] `col_index` exceeds matrix size:\n";
            error_msg += "\tcolIndex = " + std::to_string(col_index) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (vec.size() != nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::setColumn] Size of `vec` is not the same as number of rows in `this`:\n";
            error_msg += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        for (auto i = 0; i < nrows_; i++) {
            operator()(i, col_index) = vec[i];
        }

    }

    /**
     * @brief Permutes blocks of a matrix according to index sets `I` and `J` for blocks of sizes specified in `R` and `C`
     * 
     * @param I Index set of permuted row indices for each block
     * @param J Index set of permuted column indices for each block
     * @param R Vector containing number of rows in each block
     * @param C Vector containing number of columns in each block
     * @return Matrix<NumericalType> Permuted matrix
     */
    Matrix<NumericalType> blockPermute(Vector<int> I, Vector<int> J, Vector<int> R, Vector<int> C) {
        std::size_t rowCheck = 0;
        for (auto r = 0; r < R.size(); r++) rowCheck += R[r];
        if (rowCheck != nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::blockPermute] Rows in `R` do not add up to number of rows in `this`:\n";
            error_msg += "\tSum of R = " + std::to_string(rowCheck) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        std::size_t colCheck = 0;
        for (auto c = 0; c < C.size(); c++) colCheck += C[c];
        if (colCheck != ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::blockPermute] Rows in `C` do not add up to number of rows in `this`:\n";
            error_msg += "\tSum of C = " + std::to_string(colCheck) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        Vector<int> IGlobal(nrows_);
        Vector<int> JGlobal(ncols_);

        std::size_t ICounter = 0;
        for (auto i = 0; i < I.size(); i++) {
            auto I_i = I[i];
            std::size_t r = 0;
            for (auto ii = 0; ii < I_i; ii++) r += R[ii];
            for (auto iii = r; iii < (r + R[I_i]); iii++) IGlobal[ICounter++] = iii;
        }

        std::size_t JCounter = 0;
        for (auto j = 0; j < J.size(); j++) {
            auto J_j = J[j];
            std::size_t c = 0;
            for (auto jj = 0; jj < J_j; jj++) c += C[jj];
            for (auto jjj = c; jjj < (c + C[J_j]); jjj++) JGlobal[JCounter++] = jjj;
        }

        return getFromIndexSet(IGlobal, JGlobal);

    }

    /**
     * @brief Getter function for a sub-block of A
     * 
     * @param row_index Starting row index
     * @param col_index Starting column index
     * @param row_length Row length
     * @param col_length Column length
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> getBlock(std::size_t row_index, std::size_t col_index, std::size_t row_length, std::size_t col_length) {
        if (row_index + row_length > nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::getBlock] Row size exceeds matrix size:\n";
            error_msg += "\trowIndex = " + std::to_string(row_index) + "\n";
            error_msg += "\trowLength = " + std::to_string(row_length) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (col_index + col_length > ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::getBlock] Column size exceeds matrix size:\n";
            error_msg += "\tcolIndex = " + std::to_string(col_index) + "\n";
            error_msg += "\tcolLength = " + std::to_string(col_length) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }

        std::vector<NumericalType> resData;
        resData.reserve(row_length * col_length);

        for (auto i = row_index; i < row_index + row_length; i++) {
            for (auto j = col_index; j < col_index + col_length; j++) {
                resData.emplace_back(operator()(i,j));
            }
        }
        return {row_length, col_length, std::move(resData)};
    }

    /**
     * @brief Setter for a sub-block of A
     * 
     * @param row_index Starting row index
     * @param col_index Starting column index
     * @param mat Sub-matrix to set
     */
    void setBlock(std::size_t row_index, std::size_t col_index, Matrix<NumericalType>& mat) {
        if (row_index + mat.nrows() > nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::setBlock] Row size exceeds matrix size:\n";
            error_msg += "\trowIndex = " + std::to_string(row_index) + "\n";
            error_msg += "\tmat.nrows() = " + std::to_string(mat.nrows()) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (col_index + mat.ncols() > ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::setBlock] Column size exceeds matrix size:\n";
            error_msg += "\tcolIndex = " + std::to_string(col_index) + "\n";
            error_msg += "\tmat.ncols() = " + std::to_string(mat.ncols()) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }

        for (auto i = row_index; i < row_index + mat.nrows(); i++) {
            for (auto j = col_index; j < col_index + mat.ncols(); j++) {
                operator()(i,j) = mat(i - row_index, j - col_index);
            }
        }
    }

    /**
     * @brief Returns a new matrix that is the transpose of this matrix
     * 
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> T() {
        Matrix<NumericalType> res(ncols_, nrows_);
        for (auto j = 0; j < ncols_; j++) {
            for (auto i = 0; i < nrows_; i++) {
                res(j, i) = data_[flattenIndex_(i, j)];
            }
        }
        return res;
    }

    /**
     * @brief Prints the matrix to an ostream
     * 
     * @param os Output stream
     * @param A Matrix
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, Matrix<NumericalType>& A) {
        os << "  [" << A.nrows() << " x " << A.ncols() << "]  " << std::endl;
        for (auto i = 0; i < A.nrows(); i++) {
            for (auto j = 0; j < A.ncols(); j++) {
                if (fabs(A(i,j)) < 1e-14) {
                    os << std::setprecision(4) << std::setw(12) << 0;    
                }
                else {
                    os << std::setprecision(4) << std::setw(12) << A(i,j);
                }
            }
            os << std::endl;
        }
        return os;
    }

    /**
     * @brief Negates this matrix
     * 
     * @return Matrix<NumericalType> 
     */
    Matrix<NumericalType> operator-() {
        Matrix<NumericalType> res(nrows_, ncols_);
        for (auto i = 0; i < res.nrows(); i++) {
            for (auto j = 0; j < res.ncols(); j++) {
                res(i,j) = -operator()(i,j);
            }
        }
        return res;
    }

    /**
     * @brief Performs the additive update of this matrix and another matrix
     * 
     * @param rhs RHS matrix
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator+=(Matrix<NumericalType>& rhs) {
        if (rhs.nrows() != nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::operator+=] Number of rows in `rhs` is not equal to number of rows in `this`:";
            error_msg += "\trhs.nrows = " + std::to_string(rhs.nrows()) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }
        if (rhs.ncols() != ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::operator+=] Number of columns in `rhs` is not equal to number of columns in `this`:";
            error_msg += "\trhs.ncols = " + std::to_string(rhs.ncols()) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        for (auto i = 0; i < nrows_; i++) {
            for (auto j = 0; j < ncols_; j++) {
                operator()(i, j) += rhs(i, j);
            }
        }
        return *this;
    }

    /**
     * @brief Performs the additive update of this matrix and a scaler
     * 
     * @param rhs RHS scaler
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator+=(NumericalType rhs) {
        for (auto i = 0; i < nrows_; i++) {
            for (auto j = 0; j < ncols_; j++) {
                operator()(i, j) += rhs;
            }
        }
        return *this;
    }

    /**
     * @brief Performs the negative update for this matrix and another matrix
     * 
     * @param rhs RHS matrix
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator-=(Matrix<NumericalType>& rhs) {
        if (rhs.nrows() != nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::operator-=] Number of rows in `rhs` is not equal to number of rows in `this`:";
            error_msg += "\trhs.nrows = " + std::to_string(rhs.nrows()) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }
        if (rhs.ncols() != ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::operator-=] Number of columns in `rhs` is not equal to number of columns in `this`:";
            error_msg += "\trhs.ncols = " + std::to_string(rhs.ncols()) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::invalid_argument(error_msg);
        }

        for (auto i = 0; i < nrows_; i++) {
            for (auto j = 0; j < ncols_; j++) {
                operator()(i, j) -= rhs(i, j);
            }
        }
        return *this;
    }

    /**
     * @brief Performs the negative update of this matrix and a scaler
     * 
     * @param rhs RHS scaler
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator-=(NumericalType rhs) {
        for (auto i = 0; i < nrows_; i++) {
            for (auto j = 0; j < ncols_; j++) {
                operator()(i, j) -= rhs;
            }
        }
        return *this;
    }

    /**
     * @brief Performs the multiplicative update for this matrix and a scaler
     * 
     * @param rhs RHS scaler
     * @return Matrix<NumericalType>& 
     */
    Matrix<NumericalType>& operator*=(NumericalType rhs) {
        for (auto i = 0; i < nrows_; i++) {
            for (auto j = 0; j < ncols_; j++) {
                operator()(i, j) += rhs;
            }
        }
        return *this;
    }

private:

    /**
     * @brief Helper function to flatten a row and column index to a global index
     * 
     * @param i Row index
     * @param j Column index
     * @return std::size_t 
     */
    std::size_t flattenIndex_(std::size_t i, std::size_t j) {
        if (i < 0 || i > nrows_) {
            std::string error_msg = "[EllipticForest::Matrix::flattenIndex] Index `i` is outside of range:\n";
            error_msg += "\ti = " + std::to_string(i) + "\n";
            error_msg += "\tnrows = " + std::to_string(nrows_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        if (j < 0 || j > ncols_) {
            std::string error_msg = "[EllipticForest::Matrix::flattenIndex] Index `j` is outside of range:\n";
            error_msg += "\tj = " + std::to_string(j) + "\n";
            error_msg += "\tncols = " + std::to_string(ncols_) + "\n";
            std::cerr << error_msg << std::endl;
            throw std::out_of_range(error_msg);
        }
        return i*ncols_ + j;
    }

};

/**
 * @brief Plus operator for two matrices
 * 
 * @tparam NumericalType 
 * @param A Left matrix
 * @param B Right matrix
 * @return Matrix<NumericalType> 
 */
template<typename NumericalType>
Matrix<NumericalType> operator+(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    Matrix<NumericalType> C = A;
    C += B;
    return C;
}

/**
 * @brief Minus operator for two matrices
 * 
 * @tparam NumericalType 
 * @param A Left matrix
 * @param B Right matrix
 * @return Matrix<NumericalType> 
 */
template<typename NumericalType>
Matrix<NumericalType> operator-(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    Matrix<NumericalType> C = A;
    C -= B;
    return C;
}

/**
 * @brief Multiplication operator for a scaler and a matrix
 * 
 * @tparam NumericalType 
 * @param a Left scaler
 * @param A Right matrix
 * @return Matrix<NumericalType> 
 */
template<typename NumericalType>
Matrix<NumericalType> operator*(NumericalType a, Matrix<NumericalType>& A) {
    Matrix<NumericalType> res(A.nrows(), A.ncols());
    for (auto i = 0; i < res.nrows(); i++) {
        for (auto j = 0; j < res.ncols(); j++) {
            res(i,j) = a * A(i,j);
        }
    }
    return res;
}

/**
 * @brief Multiplication operator for a matrix and a vector
 * 
 * Wraps LAPACK `dgemv`.
 * 
 * @param A Left matrix
 * @param x Right vector
 * @return Vector<double> 
 */
static Vector<double> operator*(Matrix<double>& A, Vector<double>& x) {
    if (A.ncols() != x.size()) {
        std::string error_msg = "[EllipticForest::Matrix::operator*] Invalid matrix and vector dimensions; `A.ncols() != x.size()`:\n";
        error_msg += "\tA.nrows = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tA.ncols = " + std::to_string(A.ncols()) + "\n";
        error_msg += "\tx.size = " + std::to_string(x.size()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }

    Vector<double> b(A.nrows());

    char TRANS_ = 'C';
    int M_ = A.ncols();
    int N_ = A.nrows();
    double ALPHA_ = 1.0;
    double* A_ = A.dataPointer();
    int LDA_ = A.ncols();
    double* X_ = x.dataPointer();
    int INCX_ = 1;
    double BETA_ = 0.0;
    double* Y_ = b.dataPointer();
    int INCY_ = 1;

    dgemv_(&TRANS_, &M_, &N_, &ALPHA_, A_, &LDA_, X_, &INCX_, &BETA_, Y_, &INCY_);

    return b;
}

/**
 * @brief Multiplication operator for two matrices
 * 
 * Wraps LAPACK `dgemm`.
 * 
 * @param A Left matrix
 * @param B Right matrix
 * @return Matrix<double> 
 */
static Matrix<double> operator*(Matrix<double>& A, Matrix<double>& B) {
    if (A.ncols() != B.nrows()) {
        std::string error_msg = "[EllipticForest::Matrix::operator*=] Invalid matrix dimensions, `A` must have same number of columnes as rows in `B`:\n";
        error_msg += "\tA = [" + std::to_string(A.nrows()) + ", " + std::to_string(A.ncols()) + "]\n";
        error_msg += "\tB = [" + std::to_string(B.nrows()) + ", " + std::to_string(B.ncols()) + "]\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }
    
    Matrix<double> CT(B.ncols(), A.nrows(), 0);

    // Setup call
    char TRANSA_ = 'C';
    char TRANSB_ = 'C';
    int M_ = A.nrows();
    int N_ = B.ncols();
    int K_ = A.ncols(); // B.cols();
    double ALPHA_ = 1.0;
    double* A_ = A.dataPointer();
    int LDA_ = K_;
    double* B_ = B.dataPointer();
    int LDB_ = N_;
    double BETA_ = 0.0;
    double* C_ = CT.dataPointer();
    int LDC_ = M_;
    dgemm_(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_, &BETA_, C_, &LDC_);

	return CT.T();
}

/**
 * @brief Linear solver for a matrix and a vector
 * 
 * Wraps LAPACK `dgesv`.
 * 
 * @param A System matrix
 * @param b RHS vector
 * @return Vector<double> 
 */
static Vector<double> solve(Matrix<double>& A, Vector<double>& b) {
    if (A.nrows() != A.ncols()) {
        std::string error_msg = "[EllipticForest::Matrix::solve] Matrix `A` is not square:\n";
        error_msg += "\tnrows = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tncols = " + std::to_string(A.ncols()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }
    if (A.nrows() != b.size()) {
        std::string error_msg = "[EllipticForest::Matrix::solve] Matrix `A` and vector `b` are not the correct size:\n";
        error_msg += "\tA.nrows() = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tb.size() = " + std::to_string(b.size()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }

    Vector<double> x(b);
    Matrix<double> AT = A.T();
    Vector<int> p(b.size());

    // Setup call
    int N_ = AT.nrows();
    int NRHS_ = 1;
    double* A_ = AT.dataPointer();
    int LDA_ = AT.nrows();
    int* IPIV_ = p.dataPointer();
    double* B_ = x.dataPointer();
    int LDB_ = b.size();
    int INFO_;
    dgesv_(&N_, &NRHS_, A_, &LDA_, IPIV_, B_, &LDB_, &INFO_);

    // Check output
    if (INFO_) {
        std::cerr << "[EllipticForest::Matrix::solve] Fortran call to `dgesv_` returned non-zero flag of: " << INFO_ << std::endl;
    }

    return x;

}

/**
 * @brief Linear solver for two matrices
 * 
 * Wraps LAPACK `dgesv`.
 * 
 * @param A System matrix
 * @param B RHS matrix
 * @return Matrix<double> 
 */
static Matrix<double> solve(Matrix<double>& A, Matrix<double>& B) {
    if (A.nrows() != A.ncols()) {
        std::string error_msg = "[EllipticForest::Matrix::solve] Matrix `A` is not square:\n";
        error_msg += "\tnrows = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tncols = " + std::to_string(A.ncols()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }
    if (A.nrows() != B.nrows()) {
        std::string error_msg = "[EllipticForest::Matrix::solve] Matrix `A` and `B` are not the correct size:\n";
        error_msg += "\tA.nrows() = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tB.nrows() = " + std::to_string(B.nrows()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }

    Matrix<double> X = B.T();
    Matrix<double> AT = A.T();
    Vector<int> p(B.nrows());

    // Setup call
    int N_ = AT.nrows();
    int NRHS_ = B.ncols();
    double* A_ = AT.dataPointer();
    int LDA_ = AT.nrows();
    int* IPIV_ = p.dataPointer();
    double* B_ = X.dataPointer();
    int LDB_ = B.nrows();
    int INFO_;
    dgesv_(&N_, &NRHS_, A_, &LDA_, IPIV_, B_, &LDB_, &INFO_);

    // Check output
    if (INFO_) {
        std::cerr << "[EllipticForest::Matrix::solve] Fortran call to `dgesv_` returned non-zero flag of: " << INFO_ << std::endl;
    }

    return X.T();

}

/**
 * @brief Creates a block diagonal matrix
 * 
 * @tparam NumericalType 
 * @param diag Vector of matrices to place on the diagonal of output matrix
 * @return Matrix<NumericalType> 
 */
template<typename NumericalType>
Matrix<NumericalType> blockDiagonalMatrix(std::vector<Matrix<NumericalType>> diag) {

    std::size_t nrows_total = 0;
    std::size_t ncols_total = 0;
    for (auto& d : diag) {
        nrows_total += d.nrows();
        ncols_total += d.ncols();
    }

    Matrix<NumericalType> res(nrows_total, ncols_total, 0);

    std::size_t row_index = 0;
    std::size_t col_index = 0;
    for (auto& d : diag) {
        res.setBlock(row_index, col_index, d);
        row_index += d.nrows();
        col_index += d.ncols();
    }
    return res;

}

/**
 * @brief Returns the infinity norm of a the difference of two matrices
 * 
 * @tparam NumericalType 
 * @param A Left matrix
 * @param B Right matrix
 * @return double 
 */
template<typename NumericalType>
double matrixInfNorm(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    if (A.nrows() != B.nrows()) {
        std::string error_msg = "[EllipticForest::Matrix::matrixInfNorm] Number of rows in `A` is not equal to number of rows in `B`:";
        error_msg += "\tA.nrows = " + std::to_string(A.nrows()) + "\n";
        error_msg += "\tB.nrows = " + std::to_string(B.nrows()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }
    if (A.ncols() != B.ncols()) {
        std::string error_msg = "[EllipticForest::Matrix::matrixInfNorm] Number of columns in `A` is not equal to number of columns in `B`:";
        error_msg += "\tA.ncols = " + std::to_string(A.ncols()) + "\n";
        error_msg += "\tB.ncols = " + std::to_string(B.ncols()) + "\n";
        std::cerr << error_msg << std::endl;
        throw std::invalid_argument(error_msg);
    }

    double maxDiff = 0;
    for (auto i = 0; i < A.nrows(); i++) {
        for (auto j = 0; j < B.ncols(); j++) {
            maxDiff = fmax(maxDiff, fabs(A(i,j) - B(i,j)));
        }
    }
    return maxDiff;

}

namespace MPI {

/**
 * @brief Wrapper of MPI_Send for Matrix<T>
 * 
 * @tparam T 
 * @param matrix Matrix to send
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm Communicator
 * @return int 
 */
template<class T>
int send(Matrix<T>& matrix, int dest, int tag, MPI::Communicator comm) {
    int rows, cols;
    rows = static_cast<int>(matrix.nrows());
    cols = static_cast<int>(matrix.ncols());
    send(rows, dest, tag+1, comm);
    send(cols, dest, tag+2, comm);
    send(matrix.dataNoConst(), dest, tag, comm);
    return 0;
}

/**
 * @brief Wrapper of MPI_Recv for Matrix<T>
 * 
 * @tparam T 
 * @param matrix Matrix to receive
 * @param src Source rank
 * @param tag Message tag
 * @param comm Communicator
 * @param status Message status
 * @return int 
 */
template<class T>
int receive(Matrix<T>& matrix, int src, int tag, MPI::Communicator comm, MPI::Status* status) {
    int rows, cols;
    std::vector<T> vec;
    receive(rows, src, tag+1, comm, status);
    receive(cols, src, tag+2, comm, status);
    receive(vec, src, tag, comm, status);
    matrix = Matrix<T>(rows, cols, vec);
    return 0;
}

/**
 * @brief Wrapper of MPI_Bcast for Matrix<T>
 * 
 * @tparam T 
 * @param matrix Matrix to broadcast
 * @param root Root rank
 * @param comm Communicator
 * @return int 
 */
template<class T>
int broadcast(Matrix<T>& matrix, int root, MPI::Communicator comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    int rows, cols;
    std::vector<T> vec;
    if (rank == root) {
        rows = static_cast<int>(matrix.nrows());
        cols = static_cast<int>(matrix.ncols());
        vec = matrix.dataNoConst();
    }
    broadcast(rows, root, comm);
    broadcast(cols, root, comm);
    broadcast(vec, root, comm);
    if (rank != root) matrix = Matrix<T>(rows, cols, vec);
    return 0;
}

} // NAMESPACE : MPI

template<typename NumericalType>
class ParallelMatrix : public MPI::MPIObject {

protected:

    int local_rows = 0;
    int local_cols = 0;
    int global_rows = 0;
    int global_cols = 0;
    bool is_created = false;

public:

    Petsc::Mat mat = NULL;

    ParallelMatrix() :
        MPIObject(MPI_COMM_WORLD)
            {}

    ParallelMatrix(MPI::Communicator comm) :
        MPIObject(comm)
            {}

    ParallelMatrix(MPI::Communicator comm, Petsc::Mat mat) :
        MPIObject(comm) {

        // Take control of mat
        this->mat = mat;
        MatGetLocalSize(mat, &local_rows, &local_cols);
        MatGetSize(mat, &global_rows, &global_cols);
        is_created = true;

    }

    ParallelMatrix(MPI::Communicator comm, int local_rows, int local_cols, int global_rows, int global_cols) :
        MPIObject(comm),
        local_rows(local_rows),
        local_cols(local_cols),
        global_rows(global_rows),
        global_cols(global_cols) {

        // Build default Petsc matrix from options
        create();
        setSizes(local_rows, local_cols, global_rows, global_cols);
        setFromOptions();
    
    }

    ParallelMatrix(MPI::Communicator comm, int local_rows, int local_cols, int global_rows, int global_cols, Petsc::MatType matrix_type) :
        MPIObject(comm),
        local_rows(local_rows),
        local_cols(local_cols),
        global_rows(global_rows),
        global_cols(global_cols) {

        // Build default Petsc matrix from options
        create();
        setSizes(local_rows, local_cols, global_rows, global_cols);
        setType(matrix_type);
    
    }

    ParallelMatrix(MPI::Communicator comm, Matrix<NumericalType>& serial_matrix) :
        MPIObject(comm),
        local_rows(PETSC_DECIDE),
        local_cols(PETSC_DECIDE),
        global_rows(serial_matrix.nrows()),
        global_cols(serial_matrix.ncols()) {

        // Create parallel matrix
        create();
        setSizes(local_rows, local_cols, global_rows, global_cols);
        setFromOptions();

        // Fill parallel matrix
        setValues(vectorRange(0, global_rows-1), vectorRange(0, global_cols-1), serial_matrix, INSERT_VALUES);

    }

    ParallelMatrix(MPI::Communicator comm, Matrix<NumericalType>& serial_matrix, Petsc::MatType matrix_type) :
        MPIObject(comm),
        local_rows(PETSC_DECIDE),
        local_cols(PETSC_DECIDE),
        global_rows(serial_matrix.nrows()),
        global_cols(serial_matrix.ncols()) {

        // Create parallel matrix
        create();
        setSizes(local_rows, local_cols, global_rows, global_cols);
        setType(matrix_type);

        // Fill parallel matrix
        setValues(vectorRange(0, global_rows-1), vectorRange(0, global_cols-1), serial_matrix, INSERT_VALUES);

    }

    // Move `sub_matrix` to `new_comm`
    ParallelMatrix(MPI::Communicator new_comm, ParallelMatrix<NumericalType>& sub_matrix) :
        MPIObject(new_comm),
        local_rows(sub_matrix.local_rows),
        local_cols(sub_matrix.local_cols),
        global_rows(sub_matrix.global_rows),
        global_cols(sub_matrix.global_cols) {

        // //
        // IS index_set_row;
        // Vector<int> idx = vectorRange(0, sub_matrix.global_rows-1);
        // ISCreateGeneral(new_comm, sub_matrix.global_rows, idx.dataPointer(), PETSC_COPY_VALUES, &index_set_row);

        // // 
        // is_created = true;
        // MatCreateSubMatrix(sub_matrix.mat, index_set_row, NULL, MAT_INITIAL_MATRIX, &mat);
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        Mat* submat;
        IS irow[1], icol[1] = {NULL};
        int rfirst, rlast;
        bool first_time = true;
        // app.log("HERE 1");
        MatGetOwnershipRange(sub_matrix.mat, &rfirst, &rlast);
        // app.log("HERE 2");
        // app.log("sub_matrix comm = " + MPI::communicatorGetName(sub_matrix.getComm()));
        ISCreateStride(sub_matrix.getComm(), rlast - rfirst, rfirst, 1, &irow[0]);
        // app.log("HERE 3");
        ISCreateStride(sub_matrix.getComm(), rlast - rfirst, rfirst, 1, &icol[0]);
        // app.log("HERE 4");
        if (first_time) {
            MatCreateSubMatrices(sub_matrix.mat, 1, irow, icol, MAT_INITIAL_MATRIX, &submat);
            MatCreateMPIMatConcatenateSeqMat(new_comm, submat[0], PETSC_DECIDE, MAT_INITIAL_MATRIX, &mat);
            first_time = false;
        }
        else {
            MatCreateSubMatrices(sub_matrix.mat, 1, irow, icol, MAT_REUSE_MATRIX, &submat);
            MatCreateMPIMatConcatenateSeqMat(new_comm, submat[0], PETSC_DECIDE, MAT_REUSE_MATRIX, &mat);
        }

    }

    ParallelMatrix(MPI::Communicator comm, ParallelMatrix<NumericalType>& matrix, IS is_row, IS is_col) :
        MPIObject(comm) {

        MatCreateSubMatrix(matrix.mat, is_row, is_col, MAT_INITIAL_MATRIX, &mat);
        MatGetLocalSize(mat, &local_rows, &local_cols);
        MatGetSize(mat, &global_rows, &global_cols);
        is_created = true;

    }

    ParallelMatrix(const ParallelMatrix& other) :
        MPIObject(other.getComm()),
        local_rows(other.local_rows),
        local_cols(other.local_cols),
        global_rows(other.global_rows),
        global_cols(other.global_cols),
        is_created(other.is_created),
        mat(other.mat)
            {}

    ParallelMatrix& operator=(ParallelMatrix&& other) {
        if (this != &other) {
            MPIObject::operator=(std::move(other));
            local_rows = other.local_rows;
            local_cols = other.local_cols;
            global_rows = other.global_rows;
            global_cols = other.global_cols;
            mat = other.mat;
            is_created = other.is_created;

            other.local_rows = 0;
            other.local_cols = 0;
            other.global_rows = 0;
            other.global_cols = 0;
            other.mat = nullptr;
            other.is_created = false;
        }
        return *this;
    }

    ~ParallelMatrix() {
        // printf("[RANK %i/%i] Calling ParallelMatrix destructor.\n", this->getRank(), this->getSize());
        if (is_created) {
            // printf("[RANK %i/%i] Destroying Mat...\n", this->getRank(), this->getSize());
            MatDestroy(&mat);
        }

        // if (raw_data != nullptr) {
        //     delete raw_data;
        // }
    }

    int localRows() const {
        return local_rows;
    }

    int localCols() const {
        return local_cols;
    }

    int globalRows() const {
        return global_rows;
    }

    int globalCols() const {
        return global_cols;
    }

    Petsc::ErrorCode create() {
        is_created = true;
        return MatCreate(this->getComm(), &mat);
    }

    Petsc::ErrorCode setSizes(int local_rows, int local_cols, int global_rows, int global_cols) {
        return MatSetSizes(mat, local_rows, local_cols, global_rows, global_cols);
    }

    Petsc::ErrorCode setType(Petsc::MatType matrix_type) {
        return MatSetType(mat, matrix_type);
    }

    Petsc::ErrorCode setFromOptions() {
        return MatSetFromOptions(mat);
    }

    Petsc::ErrorCode setValue(int row_index, int col_index, NumericalType value, Petsc::InsertMode mode) {
        return MatSetValue(mat, row_index, col_index, value, mode);
    }

    Petsc::ErrorCode setValues(Vector<int> row_indices, Vector<int> col_indices, Matrix<NumericalType> values, Petsc::InsertMode mode) {
        int m = row_indices.size();
        int n = col_indices.size();
        return MatSetValues(mat, m, row_indices.data().data(), n, col_indices.data().data(), values.data().data(), mode);
    }

    Petsc::ErrorCode setValues(Vector<int> row_indices, Vector<int> col_indices, Vector<NumericalType> values, Petsc::InsertMode mode) {
        return MatSetValues(mat, row_indices.size(), row_indices.data().data(), col_indices.size(), col_indices.data().data(), values.data().data(), mode);
    }

    Petsc::ErrorCode getValue(int row_index, int col_index, NumericalType& value) {
        return MatGetValue(mat, row_index, col_index, &value);
    }

    Petsc::ErrorCode getValues(Vector<int> row_indices, Vector<int> col_indices, Matrix<NumericalType>& values) {
        return MatGetValues(mat, row_indices.size(), row_indices.data().data(), col_indices.size(), col_indices.data().data(), values.dataPointer());
    }

    Petsc::ErrorCode getValues(Vector<int> row_indices, Vector<int> col_indices, Vector<NumericalType>& values) {
        return MatGetValues(mat, row_indices.size(), row_indices.data().data(), col_indices.size(), col_indices.data().data(), values.dataPointer());
    }

    Petsc::ErrorCode beginAssembly(Petsc::MatAssemblyType assembly_type) {
        return MatAssemblyBegin(mat, assembly_type);
    }

    Petsc::ErrorCode endAssembly(Petsc::MatAssemblyType assembly_type) {
        return MatAssemblyEnd(mat, assembly_type);
    }

};

} // NAMESPACE : EllipticForest

#endif // MATRIX_HPP_