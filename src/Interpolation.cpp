#include "Interpolation.hpp"

namespace EllipticForest {

InterpolantBase::InterpolantBase(Vector<double>& x, Vector<double>& y, int m) :
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

LinearInterpolant::LinearInterpolant(Vector<double>& x, Vector<double>& y) :
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

PolynomialInterpolant::PolynomialInterpolant(Vector<double>& x, Vector<double>& y, int poly_order) :
    InterpolantBase(x, y, poly_order+1),
    dy(0.)
        {}

double PolynomialInterpolant::rawInterp(int j, double x) {
    int i, m, ns=0;
    double y, den, dif, dift, ho, hp, w;
    const double *xa = &xx[j], *ya = &yy[j];
    Vector<double> c(mm);
    Vector<double> d(mm);
    dif = abs(x - xa[0]);
    for (i = 0; i < mm; i++) {
        if ((dift = abs(x - xa[i])) < dif) {
            ns = i;
            dif = dift;
        }
        c[i] = ya[i];
        d[i] = ya[i];
    }
    y = ya[ns--];
    for (m=1;m<mm;m++) {
        for (i=0;i<mm-m;i++) {
            ho=xa[i]-x;
            hp=xa[i+m]-x;
            w=c[i+1]-d[i];
            if ((den=ho-hp) == 0.0) throw("Poly_interp error");
            den=w/den;
            d[i]=hp*den;
            c[i]=ho*den;
        }
        y += (dy=(2*(ns+1) < (mm-m) ? c[ns+1] : d[ns--]));
    }
    return y; 
}

BilinearInterpolant::BilinearInterpolant(Vector<double>& x1, Vector<double>& x2, Vector<double>& y) :
    m(x1.size()),
    n(x2.size()),
    y(y),
    x1_interpolant(x1, x1),
    x2_interpolant(x2, x2)
        {}

double BilinearInterpolant::operator()(double x1, double x2) {

    int i,j;
    double yy, t, u;
    i = x1_interpolant.cor ? x1_interpolant.hunt_(x1) : x1_interpolant.locate_(x1);
    j = x2_interpolant.cor ? x2_interpolant.hunt_(x2) : x2_interpolant.locate_(x2);

    int ii0 = j + i*n;
    int ii1 = j + (i+1)*n;
    int ii2 = (j+1) + (i+1)*n;
    int ii3 = (j+1) + i*n;
    t = (x1 - x1_interpolant.xx[i])/(x1_interpolant.xx[i+1] - x1_interpolant.xx[i]);
    u = (x2 - x2_interpolant.xx[j])/(x2_interpolant.xx[j+1] - x2_interpolant.xx[j]);
    yy = (1. - t)*(1. - u)*y[ii0] + t*(1. - u)*y[ii1] + t*u*y[ii2] + (1. - t)*u*y[ii3];
    return yy;

}

Vector<double> BilinearInterpolant::operator()(Vector<double>& x1, Vector<double>& x2) {
    
    Vector<double> yy(x1.size()*x2.size());
    for (int i = 0; i < x1.size(); i++) {
        double xx1 = x1[i];
        for (int j = 0; j < x2.size(); j++) {
            double xx2 = x2[j];
            int ii = j + i*x2.size();
            yy[ii] = operator()(xx1, xx2);
        }
    }
    return yy;

}

Polynomial2DInterpolant::Polynomial2DInterpolant(Vector<double>& x1, Vector<double>& x2, Vector<double>& y, int x1_poly_order, int x2_poly_order) :
    m(x1.size()),
    n(x2.size()),
    mm(x1_poly_order+1),
    nn(x2_poly_order+1),
    y(y),
    yv(m, 0),
    x1_interpolant(x1, yv, mm),
    x2_interpolant(x2, x2, nn)
        {}

double Polynomial2DInterpolant::operator()(double x1, double x2) {
    int i, j, k;
    i = x1_interpolant.cor ? x1_interpolant.hunt_(x1) : x1_interpolant.locate_(x1);
    j = x2_interpolant.cor ? x2_interpolant.hunt_(x2) : x2_interpolant.locate_(x2);
    for (k = i; k < i+mm; k++) {
        int kk = 0 + k*n;
        x2_interpolant.yy = y(kk, kk + n - 1);
        yv[k] = x2_interpolant.rawInterp(j, x2);
        // yv = x2_interpolant(x2);
    }
    return x1_interpolant.rawInterp(i, x1);
}

Vector<double> Polynomial2DInterpolant::operator()(Vector<double>& x1, Vector<double>& x2) {
    
    Vector<double> yy(x1.size()*x2.size());
    for (int i = 0; i < x1.size(); i++) {
        double xx1 = x1[i];
        for (int j = 0; j < x2.size(); j++) {
            double xx2 = x2[j];
            int ii = j + i*x2.size();
            yy[ii] = operator()(xx1, xx2);
        }
    }
    return yy;

}

}