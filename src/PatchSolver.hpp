#ifndef PATCH_SOLVER_HPP_
#define PATCH_SOLVER_HPP_

#include <string>
#include <cmath>

#include "PatchGrid.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"

namespace EllipticForest {

template<typename FloatingPointType>
class PatchSolverBase {

public:

    virtual std::string name() = 0;
    virtual Vector<FloatingPointType> solve(PatchGridBase<FloatingPointType>& grid, Vector<FloatingPointType>& dirichletData, Vector<FloatingPointType>& rhsData) = 0;
    virtual Vector<FloatingPointType> mapD2N(PatchGridBase<FloatingPointType>& grid, Vector<FloatingPointType>& dirichletData, Vector<FloatingPointType>& rhsData) = 0;
    virtual Matrix<FloatingPointType> buildD2N(PatchGridBase<FloatingPointType>& grid) = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_SOLVER_HPP_