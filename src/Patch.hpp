#ifndef PATCH_HPP_
#define PATCH_HPP_

#include <vector>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"

namespace EllipticForest {

template<typename PatchGridType, typename PatchSolverType, typename FloatingPointType>
class PatchBase {

    // Metadata
    /**
     * @brief Leaf indexed quadtree index (p4est index)
     * 
     */
    int leafID = -1;

    /**
     * @brief Globally indexed quadtree (i.e., full quadtree) index
     * 
     */
    int globalID = -1;

    /**
     * @brief Level of the patch in the quadtree
     * 
     */
    int level = -1;

    /**
     * @brief Number of times the patch has been coarsened
     * 
     */
    int nCoarsens = 0;

    /**
     * @brief Returns if the patch is a leaf or not
     * 
     * Default behavior is to check if the `leafID` is equal to -1 or not.
     * 
     * @return true 
     * @return false 
     */
    virtual bool isLeaf() { return leafID == -1 ? false : true; }

    /**
     * @brief Returns the name of the patch
     * 
     * @return std::string 
     */
    virtual std::string name() = 0;

    /**
     * @brief Returns a reference to the patch grid
     * 
     * @return PatchGridBase<FloatingPointType>& 
     */
    virtual PatchGridBase<FloatingPointType>& grid() = 0;

    // Data matrices to be formed by derived class
    /**
     * @brief Returns the data matrix X : maps load on the interior to potential on the interior
     * 
     * @return Matrix<FloatingPointType>& 
     */
    virtual Matrix<FloatingPointType>& matrixX() = 0;

    /**
     * @brief Returns the data matrix H : maps load on the interior to flux on the exterior
     * 
     * @return Matrix<FloatingPointType>& 
     */
    virtual Matrix<FloatingPointType>& matrixH() = 0;

    /**
     * @brief Returns the data matrix S (solution matrix) : maps potential on the exterior to potential on the interior
     * 
     * @return Matrix<FloatingPointType>& 
     */
    virtual Matrix<FloatingPointType>& matrixS() = 0;

    /**
     * @brief Returns the data matrix T (Dirichlet-to-Neumann matrix) : maps potential on the exterior to flux on the exterior
     * 
     * @return Matrix<FloatingPointType>& 
     */
    virtual Matrix<FloatingPointType>& matrixT() = 0;

    /**
     * @brief Returns the data vector u (solution data) : potential on the interior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorU() = 0;

    /**
     * @brief Returns the data vector g (Dirichlet data) : potential on the exterior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorG() = 0;

    /**
     * @brief Returns the data vector v (Neumann data) : flux on the exterior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorV() = 0;

    /**
     * @brief Returns the data vector f (non-homogeneous data) : load on the interior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorF() = 0;

    /**
     * @brief Returns the data vector h (particular Neumann data) : particular flux on the exterior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorH() = 0;

    /**
     * @brief Returns the data vector w (particular solution data) : particular potential on the interior
     * 
     * @return Vector<FloatingPointType>& 
     */
    virtual Vector<FloatingPointType>& vectorW() = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_HPP_