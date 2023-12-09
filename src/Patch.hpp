#ifndef PATCH_HPP_
#define PATCH_HPP_

#include <vector>

#include <petsc.h>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"

namespace EllipticForest {

template<typename DerivedType, typename PatchGridType, typename PatchSolverType, typename FloatingPointType>
class PatchBase {

public:

    /**
     * @brief Storage for the parallel matrix X
     * 
     */
    ParallelMatrix<double> par_matrix_X;

    /**
     * @brief Storage for the parallel matrix H
     * 
     */
    ParallelMatrix<double> par_matrix_H;

    /**
     * @brief Storage for the parallel matrix S
     * 
     */
    ParallelMatrix<double> par_matrix_S;

    /**
     * @brief Storage for the parallel matrix T
     * 
     */
    ParallelMatrix<double> par_matrix_T;

    /**
     * @brief Storage for the parallel vector u
     * 
     */
    ParallelVector<double> par_vector_u;

    /**
     * @brief Storage for the parallel vector g
     * 
     */
    ParallelVector<double> par_vector_g;

    /**
     * @brief Storage for the parallel vector v
     * 
     */
    ParallelVector<double> par_vector_v;

    /**
     * @brief Storage for the parallel vector f
     * 
     */
    ParallelVector<double> par_vector_f;

    /**
     * @brief Storage for the parallel vector h
     * 
     */
    ParallelVector<double> par_vector_h;

    /**
     * @brief Storage for the parallel vector w
     * 
     */
    ParallelVector<double> par_vector_w;

    /**
     * @brief Number of times the patch has been coarsened
     * 
     */
    int n_coarsens = 0;

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

    int size() {
        return grid().nx() / pow(2, n_coarsens);
    }

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