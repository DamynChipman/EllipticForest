#include "Interpolation.hpp"

namespace EllipticForest {

InterpolantBase::InterpolantBase(Vector<double>& x, const Vector<double>& y, int m) :
    n(x.size()),
    mm(m),
    jsav(0),
    cor(0),
    xx(x),
    yy(y) {
    dj = std::min(1, (int) pow((double) n, 0.25));
}

double InterpolantBase::operator()(double x) {
    int jlo = cor ? hunt_(x) : locate_(x);
    return rawInterp(jlo, x);
}

Vector<double> InterpolantBase::operator()(Vector<double>& x) {
    Vector<double> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        int jlo = cor ? hunt_(x[i]) : locate_(x[i]);
        y[i] = rawInterp(jlo, x[i]);
    }
    return y;
}

int InterpolantBase::locate_(const double x) {
    int ju, jm, jl;
    if (n < 2 || mm < 2 || mm > n) {
        throw std::invalid_argument("[EllipticForest::InterpolantBase::locate_] locate size error");
    }
    bool ascend = (xx[n-1] >= xx[0]);
    jl = 0;
    ju = n - 1;
    while (ju - jl > 1) {
        jm = (ju + jl) >> 1;
        if (x >= xx[jm] == ascend) {
            jl = jm;
        }
        else {
            ju = jm;
        }
    }
    cor = abs(jl - jsav) > dj ? 0 : 1;
    jsav = 1;
    return std::max(0, std::min(n-mm, jl-((mm-2) >> 1)));
}

int InterpolantBase::hunt_(const double x) {
    int jl = jsav, jm, ju, inc = 1;
    if (n < 2 || mm < 2 || mm > n) {
        throw std::invalid_argument("[EllipticForest::InterpolantBase::hunt_] hunt size error");
    }
    bool ascend = (xx[n-1] >= xx[0]);
    if (jl < 0 || jl > n - 1) {
        jl = 0;
        ju = n - 1;
    }
    else {
        if (x >= xx[jl] == ascend) {
            for (;;) {
                ju = jl + inc;
                if (ju >= n - 1) {
                    ju = n - 1;
                    break;
                }
                else if (x < xx[ju] == ascend) {
                    break;
                }
                else {
                    jl = ju;
                    inc += inc;
                }
            }
        }
        else {
            ju = jl;
            for (;;) {
                jl = jl - inc;
                if (jl <= 0) {
                    jl = 0;
                    break;
                }
                else if (x >= xx[jl] == ascend) {
                    break;
                }
                else {
                    ju = jl;
                    inc += inc;
                }
            }
        }
    }
    while (ju - jl > 1) {
        jm = (ju + jl) >> 1;
        if (x >= xx[jm] == ascend) {
            jl = jm;
        }
        else {
            ju = jm;
        }
    }
    cor = abs(jl - jsav) > dj ? 0 : 1;
    jsav = 1;
    return std::max(0, std::min(n - mm, jl - ((mm-2) >> 1)));
}

LinearInterpolant::LinearInterpolant(Vector<double>& x, const Vector<double>& y) :
    InterpolantBase(x, y, 2)
        {}

double LinearInterpolant::rawInterp(int j, double x) {
    if (this->xx[j] == this->xx[j+1]) {
        return this->yy[j];
    }
    else {
        return yy[j] + ((x - xx[j]) / (xx[j+1] - xx[j]))*(yy[j+1] - yy[j]);
    }
}

};