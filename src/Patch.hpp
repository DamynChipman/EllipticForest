#ifndef PATCH_HPP_
#define PATCH_HPP_

#include <vector>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"

namespace EllipticForest {

template<typename FloatingPointType>
class PatchBase {

    // Metadata
    virtual int leafID = -1;
    virtual int globalID = -1;
    virtual int level = -1;
    virtual bool isLeaf() { return leafID == -1 ? false : true; }
    virtual int nCoarsens = 0;

    // Patch grid
    PatchGridBase<FloatingPointType>* grid;

    // Data matrices to be formed by derived class
    virtual Matrix<FloatingPointType> matrixX() = 0;
    virtual Matrix<FloatingPointType> matrixH() = 0;
    virtual Matrix<FloatingPointType> matrixS() = 0;
    virtual Matrix<FloatingPointType> matrixT() = 0;

    virtual Vector<FloatingPointType> vectorU() = 0;
    virtual Vector<FloatingPointType> vectorG() = 0;
    virtual Vector<FloatingPointType> vectorV() = 0;
    virtual Vector<FloatingPointType> vectorF() = 0;
    virtual Vector<FloatingPointType> vectorH() = 0;
    virtual Vector<FloatingPointType> vectorW() = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_HPP_