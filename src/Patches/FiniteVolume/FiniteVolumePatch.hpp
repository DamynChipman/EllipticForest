#ifndef FINITE_VOLUME_PATCH_HPP_
#define FINITE_VOLUME_PATCH_HPP_

#include "FiniteVolumeGrid.hpp"
#include "FiniteVolumeSolver.hpp"
#include "../../Patch.hpp"
#include "../../EllipticForest.hpp"

namespace EllipticForest {

class FiniteVolumePatch : public MPI::MPIObject, public PatchBase<FiniteVolumePatch, FiniteVolumeGrid, FiniteVolumeSolver, double> {

protected:

    /**
     * @brief Serial X matrix from HPS method on patch
     * 
     */
    Matrix<double> X_serial{};

    /**
     * @brief Serial H matrix from HPS method on patch
     * 
     */
    Matrix<double> H_serial{};

    /**
     * @brief Serial S matrix from HPS method on patch
     * 
     */
    Matrix<double> S_serial{};

    /**
     * @brief Serial T matrix from HPS method on patch
     * 
     */
    Matrix<double> T_serial{};

    /**
     * @brief Serial u vector from HPS method on patch
     * 
     */
    Vector<double> u_serial{};

    /**
     * @brief Serial g vector from HPS method on patch
     * 
     */
    Vector<double> g_serial{};

    /**
     * @brief Serial v vector from HPS method on patch
     * 
     */
    Vector<double> v_serial{};

    /**
     * @brief Serial f vector from HPS method on patch
     * 
     */
    Vector<double> f_serial{};

    /**
     * @brief Serial h vector from HPS method on patch
     * 
     */
    Vector<double> h_serial{};

    /**
     * @brief Serial w vector from HPS method on patch
     * 
     */
    Vector<double> w_serial{};

    /**
     * @brief Parallel X matrix from HPS method on patch
     * 
     */
    ParallelMatrix<double> X_parallel{};

    /**
     * @brief Parallel H matrix from HPS method on patch
     * 
     */
    ParallelMatrix<double> H_parallel{};

    /**
     * @brief Parallel S matrix from HPS method on patch
     * 
     */
    ParallelMatrix<double> S_parallel{};

    /**
     * @brief Parallel T matrix from HPS method on patch
     * 
     */
    ParallelMatrix<double> T_parallel{};

    /**
     * @brief Parallel u vector from HPS method on patch
     * 
     */
    ParallelVector<double> u_parallel{};

    /**
     * @brief Parallel g vector from HPS method on patch
     * 
     */
    ParallelVector<double> g_parallel{};

    /**
     * @brief Parallel v vector from HPS method on patch
     * 
     */
    ParallelVector<double> v_parallel{};

    /**
     * @brief Parallel f vector from HPS method on patch
     * 
     */
    ParallelVector<double> f_parallel{};

    /**
     * @brief Parallel h vector from HPS method on patch
     * 
     */
    ParallelVector<double> h_parallel{};

    /**
     * @brief Parallel w vector from HPS method on patch
     * 
     */
    ParallelVector<double> w_parallel{};

    /**
     * @brief The grid on the patch
     * 
     */
    FiniteVolumeGrid grid_;

public:

    /**
     * @brief Construct a new FiniteVolumePatch object (default)
     * 
     */
    FiniteVolumePatch();

    /**
     * @brief Construct a new FiniteVolumePatch object on a communicator
     * 
     * @param comm MPI communicator
     */
    FiniteVolumePatch(MPI::Communicator comm);

    /**
     * @brief Construct a new FiniteVolumePatch object on a communicator with a grid
     * 
     * @param comm MPI communicator
     * @param grid Finite volume grid
     */
    FiniteVolumePatch(MPI::Communicator comm, FiniteVolumeGrid grid);

    /**
     * @brief Destroy the FiniteVolumePatch object
     * 
     */
    ~FiniteVolumePatch();

    /**
     * @brief Return the name of the patch
     * 
     * @return std::string 
     */
    virtual std::string name();

    /**
     * @brief Return a reference to the grid
     * 
     * @return FiniteVolumeGrid& 
     */
    virtual FiniteVolumeGrid& grid();

    /**
     * @brief Return a reference to the X matrix
     * 
     * @return Matrix<double>& 
     */
    virtual Matrix<double>& matrixX();

    /**
     * @brief Return a reference to the H matrix
     * 
     * @return Matrix<double>& 
     */
    virtual Matrix<double>& matrixH();

    /**
     * @brief Return a reference to the S matrix
     * 
     * @return Matrix<double>& 
     */
    virtual Matrix<double>& matrixS();

    /**
     * @brief Return a reference to the T matrix
     * 
     */
    virtual Matrix<double>& matrixT();

    /**
     * @brief Return a reference to the u vector
     * 
     */
    virtual Vector<double>& vectorU();

    /**
     * @brief Return a reference to the g vector
     * 
     */
    virtual Vector<double>& vectorG();

    /**
     * @brief Return a reference to the v vector
     * 
     */
    virtual Vector<double>& vectorV();

    /**
     * @brief Return a reference to the f vector
     * 
     */
    virtual Vector<double>& vectorF();

    /**
     * @brief Return a reference to the h vector
     * 
     */
    virtual Vector<double>& vectorH();

    /**
     * @brief Return a reference to the w vector
     * 
     */
    virtual Vector<double>& vectorW();

    /**
     * @brief Returns the size of the data stored on the patch in MB
     * 
     * @return double 
     */
    double dataSize();

    /**
     * @brief Returns a string version of the patch
     * 
     * @return std::string 
     */
    std::string str();    

};

namespace MPI {

/**
 * @brief Function overload for @sa `broadcast` for EllipticForest::FiniteVolumePatch
 *
 * @param patch The patch to communicate
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<>
int broadcast(FiniteVolumePatch& patch, int root, MPI::Communicator comm);

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_PATCH_HPP_