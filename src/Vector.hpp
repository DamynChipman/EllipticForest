#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <initializer_list>

namespace EllipticForest {

template<typename NumericalType>
class Vector {

protected:

    std::size_t size_;
    std::vector<NumericalType> data_;

public:

    // ---======---
    // Constructors
    // ---======---

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
     * @brief Create a vector and assign it to already allocated memory array
     * 
     * @param size Size of vector
     * @param dataArray Pointer of beginning of memory block
     */
    // Vector(std::size_t size, NumericalType* dataArray) : size_(size) {
    //     data_.assign(dataArray, dataArray + size);
    // }

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

    // Vector(Vector&& v) {
    //     *this = std::move(v);
    // }

    // ---=========================---
    // "Getter" and "Setter" functions
    // ---=========================---

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
            std::string errorMessage = "[EllipticForest::Vector::getEntry] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
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
            std::string errorMessage = "[EllipticForest::Vector::operator[]] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
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
            std::string errorMessage = "[EllipticForest::Vector::operator[]] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
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
            std::string errorMessage = "[EllipticForest::Vector::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return data_[index];
    }
    
    // ---============---
    // "Getter" functions
    // ---============---

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
            std::string errorMessage = "[EllipticForest::Vector::getRange] `a` or `b` is outside of range of vector:\n";
            errorMessage += "\ta = " + std::to_string(a) + "\n";
            errorMessage += "\tb = " + std::to_string(b) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
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
            std::string errorMessage = "[EllipticForest::Vector::operator()] `a` or `b` is outside of range of vector:\n";
            errorMessage += "\ta = " + std::to_string(a) + "\n";
            errorMessage += "\tb = " + std::to_string(b) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        Vector<NumericalType> v((b - a) + 1);
        for (auto i = 0; i < v.size(); i++) {
            v(i) = a + i;
        }
        return v;
    }

    Vector<NumericalType> getSegment(std::size_t startIndex, std::size_t length) {
        if (startIndex + length > size_) {
            std::string errorMessage = "[EllipticForest::Vector::getSegment] Index mismatch. `startIndex` + `length` is greater than size of vector:\n";
            errorMessage += "\tstartIndex = " + std::to_string(startIndex) + "\n";
            errorMessage += "\tlength = " + std::to_string(length) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        Vector<NumericalType> res(length);
        auto ii = 0;
        for (auto i = startIndex; i < startIndex + length; i++) {
            res[ii++] = data_[i];
        }
        return res;

    }

    void setSegment(std::size_t startIndex, const Vector<NumericalType>& vec) {
        if (startIndex + vec.size() > size_) {
            std::string errorMessage = "[EllipticForest::Vector::setSegment] Index mismatch. `startIndex` + `vec.size()` is greater than size of host vector:\n";
            errorMessage += "\tstartIndex = " + std::to_string(startIndex) + "\n";
            errorMessage += "\tvec.size() = " + std::to_string(vec.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        for (auto i = startIndex; i < startIndex + vec.size(); i++) {
            data_[i] = vec[i - startIndex];
        }
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
            std::string errorMessage = "[EllipticForest::Vector::operator()] `Size of index set `I` is greater than size of vector:\n";
            errorMessage += "\tI.size() = " + std::to_string(I.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        Vector<NumericalType> res(I.size());
        for (auto i = 0; i < I.size(); i++) {
            if (I[i] > size_ || I[i] < 0) {
                std::string errorMessage = "[EllipticForest::Vector::operator()] Index in `I` is out of range:\n";
                errorMessage += "\ti = " + std::to_string(i) + "\n";
                errorMessage += "\tI[i] = " + std::to_string(I[i]) + "\n";
                errorMessage += "\tsize = " + std::to_string(size_) + "\n";
                std::cerr << errorMessage << std::endl;
                throw std::out_of_range(errorMessage);
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
            std::string errorMessage = "[EllipticForest::Vector::operator()] `Size of index set `I` is greater than size of vector:\n";
            errorMessage += "\tI.size() = " + std::to_string(I.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }

        Vector<NumericalType> res(I.size());
        for (auto i = 0; i < I.size(); i++) {
            if (I[i] > size_ || I[i] < 0) {
                std::string errorMessage = "[EllipticForest::Vector::operator()] Index in `I` is out of range:\n";
                errorMessage += "\ti = " + std::to_string(i) + "\n";
                errorMessage += "\tI[i] = " + std::to_string(I[i]) + "\n";
                errorMessage += "\tsize = " + std::to_string(size_) + "\n";
                std::cerr << errorMessage << std::endl;
                throw std::out_of_range(errorMessage);
            }
            res(i) = operator()(I(i));
        }
        return res;
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
            std::string errorMessage = "[EllipticForest::Vector::blockPermute] Sizes in `S` do not add upt to the size of `this`:\n";
            errorMessage += "\tSum of `S` = " + std::to_string(sizeCheck) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
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

    // ---=========---
    // Math operations
    // ---=========---

    Vector<NumericalType> operator-() {
        Vector<NumericalType> res(size_);
        for (auto i = 0; i < res.size(); i++) {
            res[i] = -data_[i];
        }
        return res;
    }

    Vector<NumericalType>& operator+=(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string errorMessage = "[EllipticForest::Vector::operator+=] Size of `rhs` is not the same of `this`:\n";
            errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto i = 0; i < size_; i++) {
            data_[i] += rhs[i];
        }
        return *this;
    }

    Vector<NumericalType>& operator+=(const NumericalType rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] += rhs;
        }
        return *this;
    }

    Vector<NumericalType> operator+(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string errorMessage = "[EllipticForest::Vector::operator+] Size of `rhs` is not the same of `this`:\n";
            errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        return Vector<NumericalType>(*this) += rhs;
        
    }

    Vector<NumericalType> operator+(const NumericalType rhs) {
        return Vector<NumericalType>(*this) += rhs;
    }

    Vector<NumericalType>& operator-=(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string errorMessage = "[EllipticForest::Vector::operator-=] Size of `rhs` is not the same of `this`:\n";
            errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        for (auto i = 0; i < size_; i++) {
            data_[i] -= rhs[i];
        }
        return *this;
    }

    Vector<NumericalType>& operator-=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] -= rhs;
        }
        return *this;
    }

    Vector<NumericalType> operator-(const Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string errorMessage = "[EllipticForest::Vector::operator-] Size of `rhs` is not the same of `this`:\n";
            errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        return Vector<NumericalType>(*this) -= rhs;
        
    }

    Vector<NumericalType> operator-(const NumericalType rhs) {
        return Vector<NumericalType>(*this) -= rhs;
    }

    // Vector<NumericalType>& operator*=(const Vector<NumericalType>& rhs) {
    //     if (rhs.size() != size_) {
    //         std::string errorMessage = "[EllipticForest::Vector::operator*=] Size of `rhs` is not the same of `this`:\n";
    //         errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
    //         errorMessage += "\tsize = " + std::to_string(size_) + "\n";
    //         std::cerr << errorMessage << std::endl;
    //         throw std::invalid_argument(errorMessage);
    //     }

    //     for (auto i = 0; i < size_; i++) {
    //         data_[i] *= rhs[i];
    //     }
    //     return *this;
    // }

    Vector<NumericalType>& operator*=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] *= rhs;
        }
        return *this;
    }

    NumericalType operator*(Vector<NumericalType>& rhs) {
        if (rhs.size() != size_) {
            std::string errorMessage = "[EllipticForest::Vector::operator*] Size of `rhs` is not the same of `this`:\n";
            errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
            errorMessage += "\tsize = " + std::to_string(size_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::invalid_argument(errorMessage);
        }

        NumericalType res = 0;
        for (auto i = 0; i < size_; i++) {
            res += rhs[i] * data_[i];
        }
        return res;
    }

    // Vector<NumericalType>& operator/=(const Vector<NumericalType>& rhs) {
    //     if (rhs.size() != size_) {
    //         std::string errorMessage = "[EllipticForest::Vector::operator/=] Size of `rhs` is not the same of `this`:\n";
    //         errorMessage += "\trhs.size() = " + std::to_string(rhs.size()) + "\n";
    //         errorMessage += "\tsize = " + std::to_string(size_) + "\n";
    //         std::cerr << errorMessage << std::endl;
    //         throw std::invalid_argument(errorMessage);
    //     }

    //     for (auto i = 0; i < size_; i++) {
    //         data_[i] /= rhs[i];
    //     }
    //     return *this;
    // }

    Vector<NumericalType>& operator/=(const NumericalType& rhs) {
        for (auto i = 0; i < size_; i++) {
            data_[i] /= rhs;
        }
        return *this;
    }

};

static Vector<int> vectorRange(int start, int end) {
    int N = (end - start) + 1;
    Vector<int> res(N);
    for (int i = 0; i < N; i++) {
        res[i] = start + i;
    }
    return res;
}

template<typename NumericalType>
Vector<NumericalType> operator*(NumericalType lhs, Vector<NumericalType> rhs) {
    Vector<NumericalType> res(rhs.size());
    for (auto i = 0; i < res.size(); i++) {
        res[i] = lhs * rhs[i];
    }
    return res;
}

template<typename NumericalType>
Vector<NumericalType> operator/(Vector<NumericalType> lhs, Vector<NumericalType> rhs) {
    if (lhs.size() != rhs.size()) {
        std::string errorMessage = "[EllipticForest::Vector::operator/] Sizes of `lhs` and `rhs` are not the same:\n";
        errorMessage += "\tlhs.size = " + std::to_string(lhs.size()) + "\n";
        errorMessage += "\trhs.size = " + std::to_string(rhs.size()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
    Vector<NumericalType> res(lhs.size());
    for (auto i = 0; i < res.size(); i++) {
        res[i] = lhs[i] / rhs[i];
    }
    return res;
} 

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

template<typename NumericalType>
Vector<NumericalType> concatenate(std::initializer_list<Vector<NumericalType>> vectors) {
    std::vector<Vector<NumericalType>> v(vectors);
    return concatenate(v);
}

template<typename NumericalType>
double vectorInfNorm(Vector<NumericalType>& a, Vector<NumericalType>& b) {
    if (a.size() != b.size()) {
        std::string errorMessage = "[EllipticForest::Vector::vectorInfNorm] Sizes of `a` and `b` are not the same:\n";
        errorMessage += "\ta.size = " + std::to_string(a.size()) + "\n";
        errorMessage += "\tb.size = " + std::to_string(b.size()) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }

    double maxDiff = 0;
    for (auto i = 0; i < a.size(); i++) {
        maxDiff = fmax(maxDiff, fabs(a[i] - b[i]));
    }
    return maxDiff;

}

// template<typename NumericalType>
// Vector<NumericalType> operator*(NumericalType lhs, Vector<NumericalType> rhs) {
//     Vector<NumericalType> res(rhs.size());
//     for (auto i = 0; i < res.size(); i++) {
//         res[i] = lhs * rhs[i];
//     }
//     return res;
// }
 
} // NAMESPACE : EllipticForest

#endif // VECTOR_HPP_