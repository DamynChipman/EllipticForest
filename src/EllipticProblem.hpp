#ifndef ELLIPTIC_PROBLEM_HPP_
#define ELLIPTIC_PROBLEM_HPP_

namespace EllipticForest {

template<typename FloatingDataType>
class EllipticProblemBase {

public:

    virtual FloatingDataType lambda() = 0;
    virtual FloatingDataType u(FloatingDataType x, FloatingDataType y) = 0;
    virtual FloatingDataType f(FloatingDataType x, FloatingDataType y) = 0;
    virtual FloatingDataType dudx(FloatingDataType x, FloatingDataType y) = 0;
    virtual FloatingDataType dudy(FloatingDataType x, FloatingDataType y) = 0;

};

// template<typename FloatingDataType>
// class EllipticProblem : public EllipticProblemBase<FloatingDataType> {

// public:



// };

} // NAMESPACE : EllipticForest

#endif // ELLIPTIC_PROBLEM_HPP_