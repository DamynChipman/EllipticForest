#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <string>
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "Vector.hpp"

extern "C" {
    void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);
    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
    void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
}

namespace EllipticForest {

template<typename NumericalType>
class Matrix {

protected:

    std::size_t nRows_;
    std::size_t nCols_;
    std::vector<NumericalType> data_;

public:

    Matrix() : nRows_(0), nCols_(0), data_(0) {}

    Matrix(std::size_t nRows, std::size_t nCols) : nRows_(nRows), nCols_(nCols), data_(nRows * nCols) {}

    // Matrix(std::size_t nRows, std::size_t nCols, NumericalType* dataArray) : nRows_(nRows), nCols_(nCols) {
    //     data_.assign(dataArray, dataArray + (nCols * nRows));
    // }

    Matrix(std::size_t nRows, std::size_t nCols, NumericalType value) : nRows_(nRows), nCols_(nCols), data_(nRows * nCols, value) {}

    Matrix(std::size_t nRows, std::size_t nCols, std::vector<NumericalType> data) : nRows_(nRows), nCols_(nCols), data_(data) {}

    Matrix(std::size_t nRows, std::size_t nCols, std::initializer_list<NumericalType> iList) : nRows_(nRows), nCols_(nCols), data_(iList) {}

    Matrix(Matrix& A) {
        nRows_ = A.nRows();
        nCols_ = A.nCols();
        data_ = A.data();
    }

    Matrix(const Matrix& A) {
        nRows_ = A.nRows();
        nCols_ = A.nCols();
        data_ = A.data();
    }

    Matrix<NumericalType>& operator=(const Matrix<NumericalType>& rhs) {
        if (&rhs != this) {
            nRows_ = rhs.nRows();
            nCols_ = rhs.nCols();
            data_ = rhs.data();
            return *this;
        }
        return *this;
    }

    std::size_t nRows() const { return nRows_; }
    std::size_t nCols() const { return nCols_; }
    std::vector<NumericalType> data() const { return data_; }
    NumericalType* dataPointer() { return data_.data(); }

    NumericalType getEntry(std::size_t i, std::size_t j) {
        return data_[flattenIndex_(i,j)];
    }

    NumericalType& operator()(std::size_t i, std::size_t j) {
        return data_[flattenIndex_(i,j)];
    }

    Matrix<NumericalType> operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        Matrix<NumericalType> res((b - a) + 1, (d - c) + 1);
        for (auto i = 0; i < res.nRows(); i++) {
            for (auto j = 0; j < res.nCols(); j++) {
                res(i,j) = data_[flattenIndex_(a + i, c + j)];
            }
        }
        return res;
    }

    Vector<NumericalType> getRow(std::size_t rowIndex) {
        Vector<NumericalType> res(nCols_);
        Matrix<NumericalType> rowMatrix = operator()(rowIndex, rowIndex, 0, nCols_ - 1);
        for (auto j = 0; j < nCols_; j++) {
            res[j] = rowMatrix(0, j);
        }
        return res;
    }

    Vector<NumericalType> getCol(std::size_t colIndex) {
        Vector<NumericalType> res(nRows_);
        Matrix<NumericalType> colMatrix = operator()(0, nRows_ - 1, colIndex, colIndex);
        for (auto i = 0; i < nCols_; i++) {
            res[i] = colMatrix(i, 0);
        }
        return res;
    }

    void setRow(std::size_t rowIndex, Vector<NumericalType>& vec) {
        if (rowIndex > nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::setRow] `rowIndex` exceeds matrix size:\n";
            errorMessage += "\trowIndex = " + std::to_string(rowIndex) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (vec.size() != nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::setRow] Size of `vec` is not the same as number of columns in `this`:\n";
            errorMessage += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto j = 0; j < nCols_; j++) {
            operator()(rowIndex, j) = vec[j];
        }

    }

    void setColumn(std::size_t colIndex, Vector<NumericalType>& vec) {
        if (colIndex > nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::setColumn] `colIndex` exceeds matrix size:\n";
            errorMessage += "\tcolIndex = " + std::to_string(colIndex) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (vec.size() != nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::setColumn] Size of `vec` is not the same as number of rows in `this`:\n";
            errorMessage += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto i = 0; i < nRows_; i++) {
            operator()(i, colIndex) = vec[i];
        }

    }

    Matrix<NumericalType> getFromIndexSet(Vector<int> I, Vector<int> J) {
        if (I.size() > nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::getFromIndexSet] Size of index set `I` is greater than number of rows in `this`:\n";
            errorMessage += "\tI.size() = " + std::to_string(I.size()) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (J.size() > nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::getFromIndexSet] Size of index set `J` is greater than number of columns in `this`:\n";
            errorMessage += "\tJ.size() = " + std::to_string(J.size()) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        Matrix<NumericalType> res(I.size(), J.size());
        for (auto i = 0; i < I.size(); i++) {
            for (auto j = 0; j < J.size(); j++) {
                if (I[i] > nRows_ || I[i] < 0) {
                    std::string errorMessage = "[EllipticForest::Matrix::getFromIndexSet] Index in `I` is out of range:\n";
                    errorMessage += "\ti = " + std::to_string(i) + "\n";
                    errorMessage += "\tI[i] = " + std::to_string(I[i]) + "\n";
                    errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
                }
                if (J[j] > nCols_ || J[j] < 0) {
                    std::string errorMessage = "[EllipticForest::Matrix::getFromIndexSet] Index in `J` is out of range:\n";
                    errorMessage += "\tj = " + std::to_string(j) + "\n";
                    errorMessage += "\tJ[j] = " + std::to_string(J[j]) + "\n";
                    errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
                }
                res(i,j) = operator()(I[i], J[j]);
            }
        }
        return res;

    }

    Matrix<NumericalType> operator()(Vector<int> I, Vector<int> J) {
        return getFromIndexSet(I,J);
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
        if (rowCheck != nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::blockPermute] Rows in `R` do not add up to number of rows in `this`:\n";
            errorMessage += "\tSum of R = " + std::to_string(rowCheck) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        std::size_t colCheck = 0;
        for (auto c = 0; c < C.size(); c++) colCheck += C[c];
        if (colCheck != nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::blockPermute] Rows in `C` do not add up to number of rows in `this`:\n";
            errorMessage += "\tSum of C = " + std::to_string(colCheck) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        Vector<int> IGlobal(nRows_);
        Vector<int> JGlobal(nCols_);

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

    Matrix<NumericalType> getBlock(std::size_t rowIndex, std::size_t colIndex, std::size_t rowLength, std::size_t colLength) {
        if (rowIndex + rowLength > nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::getBlock] Row size exceeds matrix size:\n";
            errorMessage += "\trowIndex = " + std::to_string(rowIndex) + "\n";
            errorMessage += "\trowLength = " + std::to_string(rowLength) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (colIndex + colLength > nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::getBlock] Column size exceeds matrix size:\n";
            errorMessage += "\tcolIndex = " + std::to_string(colIndex) + "\n";
            errorMessage += "\tcolLength = " + std::to_string(colLength) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        std::vector<NumericalType> resData;
        resData.reserve(rowLength * colLength);

        for (auto i = rowIndex; i < rowIndex + rowLength; i++) {
            for (auto j = colIndex; j < colIndex + colLength; j++) {
                resData.emplace_back(operator()(i,j));
            }
        }
        return {rowLength, colLength, std::move(resData)};
    }

    void setBlock(std::size_t rowIndex, std::size_t colIndex, Matrix<NumericalType>& mat) {
        if (rowIndex + mat.nRows() > nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::setBlock] Row size exceeds matrix size:\n";
            errorMessage += "\trowIndex = " + std::to_string(rowIndex) + "\n";
            errorMessage += "\tmat.nRows() = " + std::to_string(mat.nRows()) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (colIndex + mat.nCols() > nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::setBlock] Column size exceeds matrix size:\n";
            errorMessage += "\tcolIndex = " + std::to_string(colIndex) + "\n";
            errorMessage += "\tmat.nCols() = " + std::to_string(mat.nCols()) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        for (auto i = rowIndex; i < rowIndex + mat.nRows(); i++) {
            for (auto j = colIndex; j < colIndex + mat.nCols(); j++) {
                operator()(i,j) = mat(i - rowIndex, j - colIndex);
            }
        }
    }

    // Matrix<NumericalType>& operator+=(const Matrix<NumericalType>& rhs) {
    //     if (rhs.nRows() != nRows_) {
    //         std::string errorMessage = "[EllipticForest::Matrix::operator+="
    //     }
    // }

    Matrix<NumericalType> T() {
        Matrix<NumericalType> res(nCols_, nRows_);
        for (auto j = 0; j < nCols_; j++) {
            for (auto i = 0; i < nRows_; i++) {
                res(j, i) = data_[flattenIndex_(i, j)];
            }
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, Matrix<NumericalType>& A) {
        os << "  [" << A.nRows() << " x " << A.nCols() << "]  " << std::endl;
        for (auto i = 0; i < A.nRows(); i++) {
            for (auto j = 0; j < A.nCols(); j++) {
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

    Matrix<NumericalType> operator-() {
        Matrix<NumericalType> res(nRows_, nCols_);
        for (auto i = 0; i < res.nRows(); i++) {
            for (auto j = 0; j < res.nCols(); j++) {
                res(i,j) = -operator()(i,j);
            }
        }
        return res;
    }

    Matrix<NumericalType>& operator+=(Matrix<NumericalType>& rhs) {
        if (rhs.nRows() != nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::operator+=] Number of rows in `rhs` is not equal to number of rows in `this`:";
            errorMessage += "\trhs.nRows = " + std::to_string(rhs.nRows()) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }
        if (rhs.nCols() != nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::operator+=] Number of columns in `rhs` is not equal to number of columns in `this`:";
            errorMessage += "\trhs.nCols = " + std::to_string(rhs.nCols()) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto i = 0; i < nRows_; i++) {
            for (auto j = 0; j < nCols_; j++) {
                operator()(i, j) += rhs(i, j);
            }
        }
        return *this;
    }

    Matrix<NumericalType>& operator+=(NumericalType rhs) {
        for (auto i = 0; i < nRows_; i++) {
            for (auto j = 0; j < nCols_; j++) {
                operator()(i, j) += rhs;
            }
        }
        return *this;
    }

    Matrix<NumericalType>& operator-=(Matrix<NumericalType>& rhs) {
        if (rhs.nRows() != nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::operator-=] Number of rows in `rhs` is not equal to number of rows in `this`:";
            errorMessage += "\trhs.nRows = " + std::to_string(rhs.nRows()) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }
        if (rhs.nCols() != nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::operator-=] Number of columns in `rhs` is not equal to number of columns in `this`:";
            errorMessage += "\trhs.nCols = " + std::to_string(rhs.nCols()) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto i = 0; i < nRows_; i++) {
            for (auto j = 0; j < nCols_; j++) {
                operator()(i, j) -= rhs(i, j);
            }
        }
        return *this;
    }

    Matrix<NumericalType>& operator-=(NumericalType rhs) {
        for (auto i = 0; i < nRows_; i++) {
            for (auto j = 0; j < nCols_; j++) {
                operator()(i, j) -= rhs;
            }
        }
        return *this;
    }

    Matrix<NumericalType>& operator*=(NumericalType rhs) {
        for (auto i = 0; i < nRows_; i++) {
            for (auto j = 0; j < nCols_; j++) {
                operator()(i, j) += rhs;
            }
        }
        return *this;
    }

private:

    std::size_t flattenIndex_(std::size_t i, std::size_t j) {
        if (i < 0 || i > nRows_) {
            std::string errorMessage = "[EllipticForest::Matrix::flattenIndex] Index `i` is outside of range:\n";
            errorMessage += "\ti = " + std::to_string(i) + "\n";
            errorMessage += "\tnRows = " + std::to_string(nRows_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        if (j < 0 || j > nCols_) {
            std::string errorMessage = "[EllipticForest::Matrix::flattenIndex] Index `j` is outside of range:\n";
            errorMessage += "\tj = " + std::to_string(j) + "\n";
            errorMessage += "\tnCols = " + std::to_string(nCols_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return i*nCols_ + j;
    }

};

template<typename NumericalType>
Matrix<NumericalType> operator+(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    Matrix<NumericalType> C = A;
    C += B;
    return C;
}

template<typename NumericalType>
Matrix<NumericalType> operator-(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    Matrix<NumericalType> C = A;
    C -= B;
    return C;
}

template<typename NumericalType>
Matrix<NumericalType> operator*(NumericalType a, Matrix<NumericalType>& A) {
    Matrix<NumericalType> res(A.nRows(), A.nCols());
    for (auto i = 0; i < res.nRows(); i++) {
        for (auto j = 0; j < res.nCols(); j++) {
            res(i,j) = a * A(i,j);
        }
    }
    return res;
}

static Vector<double> operator*(Matrix<double>& A, Vector<double>& x) {
    if (A.nCols() != x.size()) {
        std::string errorMessage = "[EllipticForest::Matrix::operator*] Invalid matrix and vector dimensions; `A.nCols() != x.size()`:\n";
        errorMessage += "\tA.nRows = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tA.nCols = " + std::to_string(A.nCols()) + "\n";
        errorMessage += "\tx.size = " + std::to_string(x.size()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }

    Vector<double> b(A.nRows());

    char TRANS_ = 'C';
    int M_ = A.nCols();
    int N_ = A.nRows();
    double ALPHA_ = 1.0;
    double* A_ = A.dataPointer();
    int LDA_ = A.nCols();
    double* X_ = x.dataPointer();
    int INCX_ = 1;
    double BETA_ = 0.0;
    double* Y_ = b.dataPointer();
    int INCY_ = 1;

    dgemv_(&TRANS_, &M_, &N_, &ALPHA_, A_, &LDA_, X_, &INCX_, &BETA_, Y_, &INCY_);

    return b;
}

static Matrix<double> operator*(Matrix<double>& A, Matrix<double>& B) {
    if (A.nCols() != B.nRows()) {
        std::string errorMessage = "[EllipticForest::Matrix::operator*=] Invalid matrix dimensions, `A` must have same number of columnes as rows in `B`:\n";
        errorMessage += "\tA = [" + std::to_string(A.nRows()) + ", " + std::to_string(A.nCols()) + "]\n";
        errorMessage += "\tB = [" + std::to_string(B.nRows()) + ", " + std::to_string(B.nCols()) + "]\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
    
    Matrix<double> CT(B.nCols(), A.nRows(), 0);

    // Setup call
    // void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, dobule* C, int* LDC);
    char TRANSA_ = 'C';
    char TRANSB_ = 'C';
    int M_ = A.nRows();
    int N_ = B.nCols();
    int K_ = A.nCols(); // B.cols();
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

static Vector<double> solve(Matrix<double>& A, Vector<double>& b) {
    if (A.nRows() != A.nCols()) {
        std::string errorMessage = "[EllipticForest::Matrix::solve] Matrix `A` is not square:\n";
        errorMessage += "\tnRows = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tnCols = " + std::to_string(A.nCols()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
    if (A.nRows() != b.size()) {
        std::string errorMessage = "[EllipticForest::Matrix::solve] Matrix `A` and vector `b` are not the correct size:\n";
        errorMessage += "\tA.nRows() = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tb.size() = " + std::to_string(b.size()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }

    Vector<double> x(b);
    Matrix<double> AT = A.T();
    Vector<int> p(b.size());

    // Setup call
    int N_ = AT.nRows();
    int NRHS_ = 1;
    double* A_ = AT.dataPointer();
    int LDA_ = AT.nRows();
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

static Matrix<double> solve(Matrix<double>& A, Matrix<double>& B) {
    if (A.nRows() != A.nCols()) {
        std::string errorMessage = "[EllipticForest::Matrix::solve] Matrix `A` is not square:\n";
        errorMessage += "\tnRows = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tnCols = " + std::to_string(A.nCols()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
    if (A.nRows() != B.nRows()) {
        std::string errorMessage = "[EllipticForest::Matrix::solve] Matrix `A` and `B` are not the correct size:\n";
        errorMessage += "\tA.nRows() = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tB.nRows() = " + std::to_string(B.nRows()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }

    Matrix<double> X = B.T();
    Matrix<double> AT = A.T();
    Vector<int> p(B.nRows());

    // Setup call
    int N_ = AT.nRows();
    int NRHS_ = B.nCols();
    double* A_ = AT.dataPointer();
    int LDA_ = AT.nRows();
    int* IPIV_ = p.dataPointer();
    double* B_ = X.dataPointer();
    int LDB_ = B.nRows();
    int INFO_;
    dgesv_(&N_, &NRHS_, A_, &LDA_, IPIV_, B_, &LDB_, &INFO_);

    // Check output
    if (INFO_) {
        std::cerr << "[EllipticForest::Matrix::solve] Fortran call to `dgesv_` returned non-zero flag of: " << INFO_ << std::endl;
    }

    return X.T();

}

template<typename NumericalType>
Matrix<NumericalType> blockDiagonalMatrix(std::vector<Matrix<NumericalType>> diag) {

    std::size_t nRowsTotal = 0;
    std::size_t nColsTotal = 0;
    for (auto& d : diag) {
        nRowsTotal += d.nRows();
        nColsTotal += d.nCols();
    }

    Matrix<NumericalType> res(nRowsTotal, nColsTotal, 0);

    std::size_t rowIndex = 0;
    std::size_t colIndex = 0;
    for (auto& d : diag) {
        res.setBlock(rowIndex, colIndex, d);
        rowIndex += d.nRows();
        colIndex += d.nCols();
    }
    return res;

}

template<typename NumericalType>
double matrixInfNorm(Matrix<NumericalType>& A, Matrix<NumericalType>& B) {
    if (A.nRows() != B.nRows()) {
        std::string errorMessage = "[EllipticForest::Matrix::matrixInfNorm] Number of rows in `A` is not equal to number of rows in `B`:";
        errorMessage += "\tA.nRows = " + std::to_string(A.nRows()) + "\n";
        errorMessage += "\tB.nRows = " + std::to_string(B.nRows()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
    if (A.nCols() != B.nCols()) {
        std::string errorMessage = "[EllipticForest::Matrix::matrixInfNorm] Number of columns in `A` is not equal to number of columns in `B`:";
        errorMessage += "\tA.nCols = " + std::to_string(A.nCols()) + "\n";
        errorMessage += "\tB.nCols = " + std::to_string(B.nCols()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }

    double maxDiff = 0;
    for (auto i = 0; i < A.nRows(); i++) {
        for (auto j = 0; j < B.nCols(); j++) {
            maxDiff = fmax(maxDiff, fabs(A(i,j) - B(i,j)));
        }
    }
    return maxDiff;

}

} // NAMESPACE : EllipticForest

#endif // MATRIX_HPP_