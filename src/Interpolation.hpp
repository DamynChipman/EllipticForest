#ifndef INTERPOLATION_HPP_
#define INTERPOLATION_HPP_

#include <algorithm>
#include "Vector.hpp"

namespace EllipticForest {

/**
 * @brief Base interpolant class used for interpolating sets of data
 * 
 * Basically copy and paste from Numerical Recipes...
 * 
 */
class InterpolantBase {

public:

    InterpolantBase(Vector<double>& x, const Vector<double>& y, int m);
    double operator()(double x);
    Vector<double> operator()(Vector<double>& x);

protected:

    int n, mm, jsav, cor, dj;
    const Vector<double>& xx;
    const Vector<double>& yy;
    virtual double rawInterp(int jlo, double x) = 0;

private:

    int locate_(const double x);
    int hunt_(const double x);

};

class LinearInterpolant : public InterpolantBase {

public:

    LinearInterpolant(Vector<double>& x, const Vector<double>& y);

protected:

    virtual double rawInterp(int j, double x);

};

} // NAMESPACE : EllipticForest

#endif // INTERPOLATION_HPP_