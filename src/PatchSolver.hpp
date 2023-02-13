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

    /**
     * @brief Returns the name of the patch solver
     * 
     * @return std::string 
     */
    virtual std::string name() = 0;

    /**
     * @brief Solves the boundary value problem on the patch
     * 
     * With the Dirichlet data on the boundary and the non-homogeneous data, solves the boundary value problem on the patch.
     * 
     * The ordering of data corresponds to the ordering in the `grid`. The ordering of the boundary data uses a WESN ordering.
     * 
     * @param grid User derived `PatchGridBase` class with the solver discretization
     * @param dirichletData Vector of Dirichlet data on the boundary of the patch at the points provided in `grid`
     * @param rhsData Vector of non-homogeneous data on the interior of the patch according to the ordering in `grid`
     * @return Vector<FloatingPointType> 
     */
    virtual Vector<FloatingPointType> solve(PatchGridBase<FloatingPointType>& grid, Vector<FloatingPointType>& dirichletData, Vector<FloatingPointType>& rhsData) = 0;

    /**
     * @brief Performs the action of the Dirichlet-to-Neumann operator by mapping Dirichlet data to Neumann data
     * 
     * Given Dirichlet data, this function returns the Neumann flux data on the boundary of the patch. The Neumann data lives on the same points as the Dirichlet data.
     * 
     * @param grid User derived `PatchGridBase` class with the solver discretization
     * @param dirichletData Vector of Dirichlet data on the boundary of the patch at the points provided in `grid`
     * @param rhsData Vector of non-homogeneous data on the interior of the patch according to the ordering in `grid`
     * @return Vector<FloatingPointType> 
     */
    virtual Vector<FloatingPointType> mapD2N(PatchGridBase<FloatingPointType>& grid, Vector<FloatingPointType>& dirichletData, Vector<FloatingPointType>& rhsData) = 0;

    /**
     * @brief Computes the explicit Dirichlet-to-Neumann matrix T
     * 
     * The Dirichlet-to-Neumann matrix depends solely on the grid discretization, thus only `grid` is provided.
     * 
     * Building the Dirichlet-to-Neumann matrix is often computationally expensive.
     * 
     * @param grid User derived `PatchGridBase` class with the solver discretization
     * @return Matrix<FloatingPointType> 
     */
    virtual Matrix<FloatingPointType> buildD2N(PatchGridBase<FloatingPointType>& grid) = 0;

    /**
     * @brief Computes the particular Neumann data needed for the non-homogeneous elliptic problem
     * 
     * @param grid User derived `PatchGridBase` class with the solver discretization
     * @param rhsData Vector of non-homogeneous data on the interior of the patch according to the ordering in `grid` 
     * @return Vector<FloatingPointType> 
     */
    virtual Vector<FloatingPointType> particularNeumannData(PatchGridBase<FloatingPointType>& grid, Vector<FloatingPointType>& rhsData) {
        Vector<FloatingPointType> gZero(2*grid.nPointsX() + 2*grid.nPointsX(), 0);
        return mapD2N(grid, gZero, rhsData);
    }

};

} // NAMESPACE : EllipticForest

#endif // PATCH_SOLVER_HPP_