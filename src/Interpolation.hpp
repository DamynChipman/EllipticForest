#ifndef INTERPOLATION_HPP_
#define INTERPOLATION_HPP_

#include <algorithm>
#include "Vector.hpp"

namespace EllipticForest {

// Forward declaration
class BilinearInterpolant;

/**
 * @brief Base interpolant class used for interpolating sets of data
 * 
 * Basically copy and paste from Numerical Recipes...
 * 
 */
class InterpolantBase {

public:

    /**
     * @brief Construct a new InterpolantBase object from `x` and `y` data with the order `m`
     * 
     * @param x Vector of x data
     * @param y Vector of y data
     * @param m Order of interpolation
     */
    InterpolantBase(Vector<double>& x, const Vector<double>& y, int m);

    /**
     * @brief Interpolate the y value from the x value
     * 
     * @param x Input x value
     * @return double 
     */
    double operator()(double x);

    /**
     * @brief Interpolate the y data from the x data
     * 
     * @param x Vector of x data
     * @return Vector<double> 
     */
    Vector<double> operator()(Vector<double>& x);

protected:

    /**
     * @brief Number of points to interpolate
     * 
     */
    int n;

    /**
     * @brief Order of interpolation
     * 
     */
    int mm;

    /**
     * @brief Saved j coordinate
     * 
     */
    int jsav;

    /**
     * @brief Something from Numerical Recipes...
     * 
     */
    int cor;

    /**
     * @brief Something from Numerical Recipes...
     * 
     */
    int dj;

    /**
     * @brief Reference to x data
     * 
     */
    const Vector<double>& xx;

    /**
     * @brief Reference to y data
     * 
     */
    const Vector<double>& yy;

    /**
     * @brief Does the actual interpolation; pure virtual function
     * 
     * @param jlo Index
     * @param x X value
     * @return double 
     */
    virtual double rawInterp(int jlo, double x) = 0;

private:

    /**
     * @brief Implementation of the locate algorithm from Numerical Recipes
     * 
     * @param x X value to locate
     * @return int 
     */
    int locate_(const double x);

    /**
     * @brief Implementation of the hunt algorithm from Numerical Recipes
     * 
     * @param x X value to hunt
     * @return int 
     */
    int hunt_(const double x);

    friend class BilinearInterpolant;

};

class LinearInterpolant : public InterpolantBase {

public:

    /**
     * @brief Construct a new LinearInterpolant object
     * 
     * @param x Vector of x values
     * @param y Vector of y values
     */
    LinearInterpolant(Vector<double>& x, const Vector<double>& y);

protected:

    /**
     * @brief Implementation of linear interpolation
     * 
     * @param j Index
     * @param x X value
     * @return double 
     */
    virtual double rawInterp(int j, double x);

};

class BilinearInterpolant {

protected:

    int m;

    int n;

    const Vector<double>& y;

    LinearInterpolant x1_interpolant;

    LinearInterpolant x2_interpolant;

public:

    BilinearInterpolant(Vector<double>& x1, Vector<double>& x2, Vector<double>& y);

    double operator()(double x1, double x2);

    Vector<double> operator()(Vector<double>& x1, Vector<double>& x2);

};

} // NAMESPACE : EllipticForest

#endif // INTERPOLATION_HPP_