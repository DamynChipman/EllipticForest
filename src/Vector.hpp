#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <string>

#include <petsc.h>
#include <petscvec.h>
#include <petscerror.h>

#include "VTK.hpp"
#include "MPI.hpp"

namespace EllipticForest {

namespace Petsc {
    using Scalar = PetscScalar;
    using Vec = Vec;
    using VecType = VecType;
    using InsertMode = InsertMode;
    using ErrorCode = PetscErrorCode;
} // NAMESPACE : Petsc

template<typename NumericalType>
class Vector : public DataArrayNodeBase, public MPI::MPIObject {

protected:

    /**
     * @brief Size of the vector
     * 
     */
    std::size_t size_;

    /**
     * @brief Backend storage of data
     * 
     */
    std::vector<NumericalType> data_;

    /**
     * @brief Name of vector (for VTK)
     * 
     */
    std::string name_ = "Vector";

    /**
     * @brief Number of components for each element in data array
     * 
     */
    std::string vtkComponents_ = "1";

    /**
     * @brief Type of data
     * 
     */
    std::string vtkType_ = "Float32";

public:

    /**
     * @brief Default constructor sets size and data to zero
     * 
     */
    Vector() : size_(0), data_(0) {}
    
    /**
     * @brief Create vector with size
     * 
     * @param size Size of vector
     */
    Vector(std::size_t size) : size_(size), data_(size) {}

    /**
     * @brief Create a vector of a given size, setting all elements to `value`
     * 
     * @param size Size of vector
     * @param value Value of data entries
     */
    Vector(std::size_t size, NumericalType value) : size_(size), data_(size, value) {}

    /**
     * @brief Create a vector from an initialization list
     * 
     * @param iList The initialization list
     */
    Vector(std::initializer_list<NumericalType> iList) : size_(iList.size()), data_(iList) {}

    /**
     * @brief Construct a from a std::vector
     * 
     * @param vec std::vector to construct from
     */
    Vector(std::vector<NumericalType> vec) : size_(vec.size()), data_(vec) {}

    /**
     * @brief Copy constructor
     * 
     * @param v 
     */
    Vector(Vector& v) {
        std::size_t sizeCopy = v.size();
        std::vector<NumericalType> dataCopy = v.data();
        size_ = sizeCopy;
        data_ = dataCopy;
    }
    
    /**
     * @brief Copy constructor
     * 
     * @param v 
     */
    Vector(const Vector& v) {
        size_ = v.size();
        data_ = v.data();
    }

    /**
     * @brief Copy assignment
     * 
     * @param rhs 
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator=(const Vector<NumericalType>& rhs) {
        if (&rhs != this) {
            size_ = rhs.size();
            data_ = rhs.data();
            return *this;
        }
        return *this;
    }

    /**
     * @brief Get an entry in the vector
     * 
     * @sa operator[]
     * @sa operator()
     *  
     * @param index Index of entry
     * @return NumericalType Value of entry
     */
    NumericalType getEntry(std::size_t index) {
        if (index > size_ || index < 0) {
            std::string error_message = "[EllipticForest::Vector::getEntry] `index` is out of range:\n";
            error_message += "\tindex = " + std::to_string(index) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }
        return data_[index];
    }

    /**
     * @brief Index into the vector
     * 
     * @sa getEntry
     * @sa operator()
     * 
     * @param index Index of entry
     * @return NumericalType& Reference to value in vector
     */
    NumericalType& operator[](std::size_t index) {
        if (index > size_ || index < 0) {
            std::string error_message = "[EllipticForest::Vector::operator[]] `index` is out of range:\n";
            error_message += "\tindex = " + std::to_string(index) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }
        return data_[index];
    }

    /**
     * @brief Index into the vector
     * 
     * @sa getEntry
     * @sa operator()
     * 
     * @param index Index of entry
     * @return NumericalType& Reference to value in vector
     */
    const NumericalType& operator[](std::size_t index) const {
        if (index > size_ || index < 0) {
            std::string error_message = "[EllipticForest::Vector::operator[]] `index` is out of range:\n";
            error_message += "\tindex = " + std::to_string(index) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }
        return data_[index];
    }

    /**
     * @brief Index into vector
     * 
     * @sa getEntry
     * @sa operator[]
     * 
     * @param index Index of entry
     * @return NumericalType& Reference to value in vector
     */
    NumericalType& operator()(std::size_t index) {
        if (index > size_ || index < 0) {
            std::string error_message = "[EllipticForest::Vector::operator()] `index` is out of range:\n";
            error_message += "\tindex = " + std::to_string(index) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }
        return data_[index];
    }

    /**
     * @brief Return the name of the vector
     * 
     * @return std::string& 
     */
    std::string& name() { return name_; }

    /**
     * @brief Return the size of vector
     * 
     * @return std::size_t Size of vector
     */
    std::size_t size() const { return size_; }

    /**
     * @brief Return the std::vector of the data
     * 
     * @return std::vector<NumericalType> Data
     */
    const std::vector<NumericalType>& data() const { return data_; }

    /**
     * @brief Return a non-const version of the backend storage vector
     * 
     * @return std::vector<NumericalType>& 
     */
    std::vector<NumericalType>& dataNoConst() { return data_; }

    /**
     * @brief Return the pointer of the data array of the std::vector
     * 
     * @return NumericalType* Pointer to data array
     */
    NumericalType* dataPointer() { return data_.data(); }

    /**
     * @brief Create a new vector of the entries in the range (inclusive)
     * 
     * @sa operator()
     * 
     * @param a Start of entries
     * @param b End of entries
     * @return Vector<NumericalType> Vector of size `(b - a) + 1` with entries
     */
    Vector<NumericalType> getRange(std::size_t a, std::size_t b) {
        if (a > size_ || b > size_ || a < 0 || b < 0) {
            std::string error_message = "[EllipticForest::Vector::getRange] `a` or `b` is outside of range of vector:\n";
            error_message += "\ta = " + std::to_string(a) + "\n";
            error_message += "\tb = " + std::to_string(b) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        Vector<NumericalType> v((b - a) + 1);
        for (auto i = 0; i < v.size(); i++) {
            v(i) = a + i;
        }
        return v;
    }

    /**
     * @brief Create a new vector of the entries in the range (inclusive)
     * 
     * @sa getRange
     * 
     * @param a Start of entries
     * @param b End of entries
     * @return Vector<NumericalType> Vector of size `(b - a) + 1` with entries
     */
    Vector<NumericalType> operator()(std::size_t a, std::size_t b) {
        if (a > size_ || b > size_ || a < 0 || b < 0) {
            std::string error_message = "[EllipticForest::Vector::operator()] `a` or `b` is outside of range of vector:\n";
            error_message += "\ta = " + std::to_string(a) + "\n";
            error_message += "\tb = " + std::to_string(b) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        Vector<NumericalType> v((b - a) + 1);
        for (auto i = 0; i < v.size(); i++) {
            v(i) = data_[a + i];
        }
        return v;
    }

    /**
     * @brief Get a segment of the vector from starting index to starting index plus length
     * 
     * @param startIndex Starting index of segment
     * @param length Length of segment
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> getSegment(std::size_t startIndex, std::size_t length) {
        if (startIndex + length > size_) {
            std::string error_message = "[EllipticForest::Vector::getSegment] Index mismatch. `startIndex` + `length` is greater than size of vector:\n";
            error_message += "\tstartIndex = " + std::to_string(startIndex) + "\n";
            error_message += "\tlength = " + std::to_string(length) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        Vector<NumericalType> res(length);
        auto ii = 0;
        for (auto i = startIndex; i < startIndex + length; i++) {
            res[ii++] = data_[i];
        }
        return res;

    }

    /**
     * @brief Set a segment of the data
     * 
     * @param startIndex Starting index of segment
     * @param vec Vector of data to set into this vector
     */
    void setSegment(std::size_t startIndex, const Vector<NumericalType>& vec) {
        if (startIndex + vec.size() > size_) {
            std::string error_message = "[EllipticForest::Vector::setSegment] Index mismatch. `startIndex` + `vec.size()` is greater than size of host vector:\n";
            error_message += "\tstartIndex = " + std::to_string(startIndex) + "\n";
            error_message += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        for (auto i = startIndex; i < startIndex + vec.size(); i++) {
            data_[i] = vec[i - startIndex];
        }
    }

    /**
     * @brief Append vector to this vector
     * 
     * @param vec Vector to append
     */
    void append(const Vector<NumericalType>& vec) {
        size_ += vec.size();
        data_.insert(data_.end(), vec.data().begin(), vec.data().end());
    }

    void append(const NumericalType& val) {
        size_ += 1;
        data_.push_back(val);
    }

    /**
     * @brief Create a new vector from an index set
     * 
     * @sa operator()
     * 
     * @param I The index set of indices
     * @return Vector<NumericalType> Vector of size `I.size()` with entries from index set
     */
    Vector<NumericalType> getFromIndexSet(Vector<int> I) {
        if (I.size() > size_) {
            std::string error_message = "[EllipticForest::Vector::operator()] `Size of index set `I` is greater than size of vector:\n";
            error_message += "\tI.size() = " + std::to_string(I.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        Vector<NumericalType> res(I.size());
        for (auto i = 0; i < I.size(); i++) {
            if (I[i] > size_ || I[i] < 0) {
                std::string error_message = "[EllipticForest::Vector::operator()] Index in `I` is out of range:\n";
                error_message += "\ti = " + std::to_string(i) + "\n";
                error_message += "\tI[i] = " + std::to_string(I[i]) + "\n";
                error_message += "\tsize = " + std::to_string(size_) + "\n";
                std::cerr << error_message << std::endl;
                throw std::out_of_range(error_message);
            }
            res(i) = operator()(I(i));
        }
        return res;
    }

    /**
     * @brief Create a new vector from an index set
     * 
     * @sa getFromIndexSet
     * 
     * @param I The index set of indices
     * @return Vector<NumericalType> Vector of size `I.size()` with entries from index set
     */
    Vector<NumericalType> operator()(Vector<int> I) {
        if (I.size() > size_) {
            std::string error_message = "[EllipticForest::Vector::operator()] `Size of index set `I` is greater than size of vector:\n";
            error_message += "\tI.size() = " + std::to_string(I.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::out_of_range(error_message);
        }

        Vector<NumericalType> res(I.size());
        for (auto i = 0; i < I.size(); i++) {
            if (I[i] > size_ || I[i] < 0) {
                std::string error_message = "[EllipticForest::Vector::operator()] Index in `I` is out of range:\n";
                error_message += "\ti = " + std::to_string(i) + "\n";
                error_message += "\tI[i] = " + std::to_string(I[i]) + "\n";
                error_message += "\tsize = " + std::to_string(size_) + "\n";
                std::cerr << error_message << std::endl;
                throw std::out_of_range(error_message);
            }
            res(i) = operator()(I(i));
        }
        return res;
    }

    void clear() {
        data_.clear();
        size_ = 0;
    }

    /**
     * @brief Permutes blocks of the vector according to I with block sizes S
     * 
     * @param I Index set of permuted row indices for each block
     * @param S Vector containing size of each block
     * @return Vector<NumericalType> New permuted vector
     */
    Vector<NumericalType> blockPermute(Vector<int> I, Vector<int> S) {
        std::size_t sizeCheck = 0;
        for (auto i = 0; i < S.size(); i++) sizeCheck += S[i];
        if (sizeCheck != size_) {
            std::string error_message = "[EllipticForest::Vector::blockPermute] Sizes in `S` do not add upt to the size of `this`:\n";
            error_message += "\tSum of `S` = " + std::to_string(sizeCheck) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        Vector<int> SGlobal(size_);

        std::size_t ICounter = 0;
        for (auto i = 0; i < I.size(); i++) {
            auto I_i = I[i];
            std::size_t s = 0;
            for (auto ii = 0; ii < I_i; ii++) s += S[ii];

            for (auto iii = s; iii < (s + S[I_i]); iii++) {
                SGlobal[ICounter++] = iii;
            }
        }

        return getFromIndexSet(SGlobal);

    }

    /**
     * @brief Write to an ostream
     * 
     * @param os Ostream reference
     * @param v Vector to write
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, Vector<NumericalType>& v) {
        os << "  [" << v.size() << "]" << std::endl;
        for (auto i = 0; i < v.size(); i++) {
            if (fabs(v[i]) < 1e-14) {
                os << std::setprecision(4) << std::setw(12) << 0;
            }
            else {
                os << std::setprecision(4) << std::setw(12) << v[i];
            }
            if (i % 8 == 7) {
                os << std::endl;
            }
        }
        return os;
    }

    /**
     * @brief Subtraction operator
     * 
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> operator-() {
        Vector<NumericalType> res(size_);
        for (auto i = 0; i < res.size(); i++) {
            res[i] = -data_[i];
        }
        return res;
    }

    /**
     * @brief Addition update operator
     * 
     * @param rhs RHS vector
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator+=(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string error_message = "[EllipticForest::Vector::operator+=] Size of `rhs` is not the same of `this`:\n";
            error_message += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        for (auto i = 0; i < size_; i++) {
            data_[i] += rhs[i];
        }
        return *this;
    }

    /**
     * @brief Addition update operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator+=(const NumericalType rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] += rhs;
        }
        return *this;
    }

    /**
     * @brief Addition operator
     * 
     * @param rhs RHS vector
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> operator+(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string error_message = "[EllipticForest::Vector::operator+] Size of `rhs` is not the same of `this`:\n";
            error_message += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        return Vector<NumericalType>(*this) += rhs;
        
    }

    /**
     * @brief Addition operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> operator+(const NumericalType rhs) {
        return Vector<NumericalType>(*this) += rhs;
    }

    /**
     * @brief Subtract update operator
     * 
     * @param rhs RHS vector
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator-=(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string error_message = "[EllipticForest::Vector::operator-=] Size of `rhs` is not the same of `this`:\n";
            error_message += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        for (auto i = 0; i < size_; i++) {
            data_[i] -= rhs[i];
        }
        return *this;
    }

    /**
     * @brief Subtract update operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator-=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] -= rhs;
        }
        return *this;
    }

    /**
     * @brief Subtract operator
     * 
     * @param rhs RHS vector
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> operator-(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string error_message = "[EllipticForest::Vector::operator-] Size of `rhs` is not the same of `this`:\n";
            error_message += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        return Vector<NumericalType>(*this) -= rhs;
        
    }

    /**
     * @brief Subtract operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType> 
     */
    Vector<NumericalType> operator-(const NumericalType rhs) {
        return Vector<NumericalType>(*this) -= rhs;
    }

    /**
     * @brief Multiplication update operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator*=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] *= rhs;
        }
        return *this;
    }

    /**
     * @brief Multiplication operator
     * 
     * @param rhs RHS vector
     * @return NumericalType 
     */
    NumericalType operator*(Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string error_message = "[EllipticForest::Vector::operator*] Size of `rhs` is not the same of `this`:\n";
            error_message += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            error_message += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << error_message << std::endl;
            throw std::invalid_argument(error_message);
        }

        NumericalType res = 0;
        for (auto i = 0; i < size_; i++) {
            res += rhs[i] * data_[i];
        }
        return res;
    }

    /**
     * @brief Division update operator
     * 
     * @param rhs RHS scalar
     * @return Vector<NumericalType>& 
     */
    Vector<NumericalType>& operator/=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] /= rhs;
        }
        return *this;
    }

    /**
     * @brief Set the type of the data in the data array
     * 
     * @param t Type
     */
    void setType(std::string t) {
        vtkType_ = t;
    }

    /**
     * @brief Get the type of data in the data array
     * 
     * @return std::string 
     */
    virtual std::string getType() {
        return vtkType_;
    }

    /**
     * @brief Get the name of the data array
     * 
     * @return std::string 
     */
    virtual std::string getName() {
        return name_;
    }

    /**
     * @brief Set the number of components per entry in the data array
     * 
     * @param n Number of components
     */
    void setNumberOfComponents(std::string n) {
        vtkComponents_ = n;
    }

    /**
     * @brief Get the number of components per entry in the data array
     * 
     * @return std::string 
     */
    virtual std::string getNumberOfComponents() {
        return vtkComponents_;
    }

    /**
     * @brief Get the format of the data array
     * 
     * @return std::string 
     */
    virtual std::string getFormat() {
        return "ascii";
    }

    /**
     * @brief Get the minimum of the data
     * 
     * @return std::string 
     */
    virtual std::string getRangeMin() {
        NumericalType min = *std::min_element(data_.begin(), data_.end());
        return std::to_string(min);
    }

    /**
     * @brief Get the maximum of the data
     * 
     * @return std::string 
     */
    virtual std::string getRangeMax() {
        NumericalType max = *std::max_element(data_.begin(), data_.end());
        return std::to_string(max);
    }

    /**
     * @brief Get the data
     * 
     * @return std::string 
     */
    virtual std::string getData() {
        std::string str = "";
        for (auto p : data_) { str += std::to_string(p) + " "; }
        return str;
    }

    int write(std::string filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return -1;
        }

        // Write the number of rows and columns as integers
        int size = data_.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));

        // Write the matrix entries in row-major format
        for (auto i = 0; i < size; i++) {
            NumericalType entry = data_[i];
            file.write(reinterpret_cast<const char*>(&entry), sizeof(NumericalType));
        }

        file.close();
        return 0;
    }

};

/**
 * @brief Create a vector of integers from [`start` to `end`)
 * 
 * @param start Starting value
 * @param end Ending value
 * @return Vector<int> 
 */
static Vector<int> vectorRange(int start, int end) {
    int N = (end - start) + 1;
    Vector<int> res(N);
    for (int i = 0; i < N; i++) {
        res[i] = start + i;
    }
    return res;
}

/**
 * @brief Multiplication operator
 * 
 * @tparam NumericalType 
 * @param lhs LHS scalar
 * @param rhs RHS vector
 * @return Vector<NumericalType> 
 */
template<typename NumericalType>
Vector<NumericalType> operator*(NumericalType lhs, Vector<NumericalType> rhs) {
    Vector<NumericalType> res(rhs.size());
    for (auto i = 0; i < res.size(); i++) {
        res[i] = lhs * rhs[i];
    }
    return res;
}

/**
 * @brief Multiplication operator (entry-wise multiplication)
 * 
 * @tparam NumericalType 
 * @param lhs LHS vector
 * @param rhs RHS vector
 * @return Vector<NumericalType> 
 */
template<typename NumericalType>
Vector<NumericalType> operator*(Vector<NumericalType> lhs, Vector<NumericalType> rhs) {
    if (lhs.size() != rhs.size()) {
        std::string error_message = "[EllipticForest::Vector::operator*] Sizes of `lhs` and `rhs` are not the same:\n";
        error_message += "\tlhs.size = " + std::to_string(lhs.size()) + "\n";
        error_message += "\trhs.size = " + std::to_string(rhs.size()) + "\n";
        std::cerr << error_message << std::endl;
        throw std::invalid_argument(error_message);
    }
    Vector<NumericalType> res(lhs.size());
    for (auto i = 0; i < res.size(); i++) {
        res[i] = lhs[i] * rhs[i];
    }
    return res;
}

/**
 * @brief Division operator (entry-wise division)
 * 
 * @tparam NumericalType 
 * @param lhs LHS vector
 * @param rhs RHS vector
 * @return Vector<NumericalType> 
 */
template<typename NumericalType>
Vector<NumericalType> operator/(Vector<NumericalType> lhs, Vector<NumericalType> rhs) {
    if (lhs.size() != rhs.size()) {
        std::string error_message = "[EllipticForest::Vector::operator/] Sizes of `lhs` and `rhs` are not the same:\n";
        error_message += "\tlhs.size = " + std::to_string(lhs.size()) + "\n";
        error_message += "\trhs.size = " + std::to_string(rhs.size()) + "\n";
        std::cerr << error_message << std::endl;
        throw std::invalid_argument(error_message);
    }
    Vector<NumericalType> res(lhs.size());
    for (auto i = 0; i < res.size(); i++) {
        res[i] = lhs[i] / rhs[i];
    }
    return res;
} 

/**
 * @brief Concatenates a vector of vectors
 * 
 * @tparam NumericalType 
 * @param vectors Vector of vectors to concatentate
 * @return Vector<NumericalType> 
 */
template<typename NumericalType>
Vector<NumericalType> concatenate(std::vector<Vector<NumericalType>> vectors) {
    std::size_t nEntries = 0;
    for (auto& v : vectors) nEntries += v.size();
    Vector<NumericalType> res(nEntries);
    std::size_t index = 0;
    for (auto& v : vectors) {
        for (auto i = 0; i < v.size(); i++) {
            res[index++] = v[i];
        }
    }
    return res;
}

/**
 * @brief Concatenates an initialization list of vectors
 * 
 * @tparam NumericalType 
 * @param vectors Initialization list
 * @return Vector<NumericalType> 
 */
template<typename NumericalType>
Vector<NumericalType> concatenate(std::initializer_list<Vector<NumericalType>> vectors) {
    std::vector<Vector<NumericalType>> v(vectors);
    return concatenate(v);
}

/**
 * @brief Computes the inf-norm of the difference of two vectors
 * 
 * @tparam NumericalType 
 * @param a LHS vector
 * @param b RHS vector
 * @return double 
 */
template<typename NumericalType>
double vectorInfNorm(Vector<NumericalType>& a, Vector<NumericalType>& b) {
    if (a.size() != b.size()) {
        std::string error_message = "[EllipticForest::Vector::vectorInfNorm] Sizes of `a` and `b` are not the same:\n";
        error_message += "\ta.size = " + std::to_string(a.size()) + "\n";
        error_message += "\tb.size = " + std::to_string(b.size()) + "\n";
        std::cerr << error_message << std::endl;
        throw std::invalid_argument(error_message);
    }

    double maxDiff = 0;
    for (auto i = 0; i < a.size(); i++) {
        maxDiff = fmax(maxDiff, fabs(a[i] - b[i]));
    }
    return maxDiff;

}

/**
 * @brief Computes the L2-norm of the difference of two vectors
 * 
 * @tparam NumericalType 
 * @param a LHS vector
 * @param b RHS vector
 * @return double 
 */
template<typename NumericalType>
double vectorL2Norm(Vector<NumericalType>& a, Vector<NumericalType>& b) {
    if (a.size() != b.size()) {
        std::string error_message = "[EllipticForest::Vector::vectorInfNorm] Sizes of `a` and `b` are not the same:\n";
        error_message += "\ta.size = " + std::to_string(a.size()) + "\n";
        error_message += "\tb.size = " + std::to_string(b.size()) + "\n";
        std::cerr << error_message << std::endl;
        throw std::invalid_argument(error_message);
    }

    double norm = 0.;
    for (int i = 0; i < a.size(); i++) {
        norm += pow(a[i] - b[i], 2);
    }
    return sqrt(norm);

}

template<typename NumericalType>
Vector<NumericalType> linspace(NumericalType a, NumericalType b, int N) {
    if (a >= b) {
        std::string error_message = "[EllipticForest::Vector::linspace] Lower limit `a` is greater than or equal to upper limit `b`:\n";
        error_message += "\ta = " + std::to_string(a) + "\n";
        error_message += "\tb = " + std::to_string(b) + "\n";
        std::cerr << error_message << std::endl;
        throw std::invalid_argument(error_message);
    }

    Vector<NumericalType> out(N);
    double dx = (b - a) / (N - 1);
    for (int i = 0; i < N; i++) {
        out[i] = i*dx + a;
    }
    return out;
}

namespace MPI {

/**
 * @brief Function overload of @sa `send` for EllipticForest::Vector<T>
 * 
 * @tparam T Type of data in vector
 * @param vector Reference to vector
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int send(Vector<T>& vector, int dest, int tag, Communicator comm) {
    return send(vector.dataNoConst(), dest, tag, comm);
}

/**
 * @brief Function overload of @sa `recieve` for EllipticForest::Vector<T>
 * 
 * @tparam T Type of data in vector
 * @param vector Reference to vector
 * @param src Source rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @param status MPI status
 * @return int 
 */
template<class T>
int receive(Vector<T>& vector, int src, int tag, Communicator comm, Status* status) {
    std::vector<T> vec;
    int res = receive(vec, src, tag, comm, status);
    vector = Vector<T>(vec);
    return res;
}

/**
 * @brief Function overload of @sa `broadcast` for EllipticForest::Vector<T>
 * 
 * @tparam T Type of data in vector
 * @param vector Reference to vector
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int broadcast(Vector<T>& vector, int root, Communicator comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    std::vector<T> vec;
    if (rank == root) vec = vector.dataNoConst();
    int res = broadcast(vec, root, comm);
    if (rank != root) vector = Vector<T>(vec);
    return res;
}

/**
 * @brief Function overload of @sa `allgather` for EllipticForest::Vector<T>
 * 
 * @tparam T Type of data in vector
 * @param send_vector Reference to vector to send
 * @param recv_vector Storage for vector to receive from involved ranks
 * @param recv_count Count of elements received from any rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgather(Vector<T>& send_vector, Vector<T>& recv_vector, int recv_count, Communicator comm) {
    return allgather(send_vector.dataNoConst(), recv_vector.dataNoConst(), recv_count, comm);
}

/**
 * @brief Function overload of @sa `allgatherv` for EllipticForest::Vector<T>
 * 
 * @tparam T Type of data in vector
 * @param send_vector Reference of vector to send
 * @param recv_vector Storage for vector to receive from involved ranks
 * @param recv_counts Vector of counts of elements received from ranks
 * @param displacements Vector of displacements of elements received from ranks to put into `recv_data`
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgatherv(Vector<T>& send_vector, Vector<T>& recv_vector, std::vector<int> recv_counts, std::vector<int> displacements, Communicator comm) {
    return allgatherv(send_vector.dataNoConst(), recv_vector.dataNoConst(), recv_counts, displacements, comm);
}

} // NAMESPACE : MPI

template<typename NumericalType>
class ParallelVector : public MPI::MPIObject {

protected:

    int local_size = 0;
    int global_size = 0;
    NumericalType* raw_data = nullptr;
    bool is_created = false;

public:
    
    Petsc::Vec vec;

    ParallelVector() :
        MPIObject(MPI_COMM_WORLD)
            {}

    ParallelVector(MPI::Communicator comm) :
        MPIObject(comm)
            {}

    ParallelVector(MPI::Communicator comm, int local_size, int global_size) :
        MPIObject(comm),
        local_size(local_size),
        global_size(global_size) {

        // Build default Petsc vector from options
        create();
        setSizes(local_size, global_size);
        setFromOptions();

    }

    ParallelVector(MPI::Communicator comm, int local_size, int global_size, Petsc::VecType vector_type) :
        MPIObject(comm),
        local_size(local_size),
        global_size(global_size) {

        // Build specific Petsc vector with specified type
        create();
        setSizes(local_size, global_size);
        setType(vector_type);

    }

    ParallelVector(MPI::Communicator comm, Vector<NumericalType>& serial_vector) :
        MPIObject(comm),
        local_size(PETSC_DECIDE),
        global_size(serial_vector.size()) {


        //
        create();
        setSizes(local_size, global_size);
        setFromOptions();

        // Fill vector with values of serial vector spread across communicator
        setValues(vectorRange(0, global_size-1), serial_vector, INSERT_VALUES);
        
    }

    ParallelVector(MPI::Communicator comm, Vector<NumericalType>& serial_vector, Petsc::VecType vector_type) :
        MPIObject(comm),
        local_size(PETSC_DECIDE),
        global_size(serial_vector.size()) {


        //
        create();
        setSizes(local_size, global_size);
        setType(vector_type);

        // Fill vector with values of serial vector spread across communicator
        setValues(vectorRange(0, global_size-1), serial_vector, INSERT_VALUES);
        
    }

    // // Move `sub_vector` to `new_comm`
    // ParallelVector(MPI::Communicator new_comm, ParallelVector<NumericalType>& sub_vector) :
    //     MPIObject(new_comm),
    //     local_size(sub_vector.local_size),
    //     global_size(sub_vector.global_size) {

    //     //


    // }

    // ParallelVector(Petsc::Vec) {

    // }

    ~ParallelVector() {
        if (is_created) {
            VecDestroy(&vec);
        }

        // if (raw_data != nullptr) {
        //     delete raw_data;
        // }
    }
    
    Petsc::ErrorCode create() {
        is_created = true;
        return VecCreate(this->getComm(), &vec);
    }

    Petsc::ErrorCode setSizes(int n, int N) {
        return VecSetSizes(vec, n, N);
    }

    Petsc::ErrorCode setType(Petsc::VecType vector_type) {
        return VecSetType(vec, vector_type);
    }

    Petsc::ErrorCode setFromOptions() {
        return VecSetFromOptions(vec);
    }

    Petsc::ErrorCode setValue(int index, NumericalType value, Petsc::InsertMode mode) {
        return VecSetValue(vec, index, value, mode);
    }

    Petsc::ErrorCode setValues(Vector<int> indices, Vector<NumericalType> values, Petsc::InsertMode mode) {
        int N = indices.size();
        return VecSetValues(vec, N, indices.data().data(), values.data().data(), mode);
    }

    Petsc::ErrorCode beginAssembly() {
        return VecAssemblyBegin(vec);
    }

    Petsc::ErrorCode endAssembly() {
        return VecAssemblyEnd(vec);
    }

};
 
} // NAMESPACE : EllipticForest

#endif // VECTOR_HPP_