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
    int leafID = -1;
    int globalID = -1;
    int level = -1;
    int nCoarsens = 0;
    virtual bool isLeaf() { return leafID == -1 ? false : true; }

    // Patch grid
    PatchGridBase<FloatingPointType>* grid;

    // Data matrices to be formed by derived class
    virtual Matrix<FloatingPointType>& matrixX() = 0;
    virtual Matrix<FloatingPointType>& matrixH() = 0;
    virtual Matrix<FloatingPointType>& matrixS() = 0;
    virtual Matrix<FloatingPointType>& matrixT() = 0;

    virtual Vector<FloatingPointType>& vectorU() = 0;
    virtual Vector<FloatingPointType>& vectorG() = 0;
    virtual Vector<FloatingPointType>& vectorV() = 0;
    virtual Vector<FloatingPointType>& vectorF() = 0;
    virtual Vector<FloatingPointType>& vectorH() = 0;
    virtual Vector<FloatingPointType>& vectorW() = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_HPP_