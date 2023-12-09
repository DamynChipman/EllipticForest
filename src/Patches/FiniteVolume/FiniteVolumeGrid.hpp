#ifndef FINITE_VOLUME_GRID_HPP_
#define FINITE_VOLUME_GRID_HPP_

#include <string>

#include <petsc.h>
#include <petscdmda.h>

#include "../../EllipticForestApp.hpp"
#include "../../MPI.hpp"
#include "../../PatchGrid.hpp"

namespace EllipticForest {

/**
 * @brief Index of dimension
 * 
 */
enum DimensionIndex {
    X = 0,
    Y = 1
};

namespace Petsc {
    using DataManagement = DM;
    using ErrorCode = PetscErrorCode;
} // NAMESPACE : Petsc

class FiniteVolumeGrid : public MPI::MPIObject, public PatchGridBase<double> {

protected:

    /**
     * @brief Number of cells in the x-direction
     * 
     */
    int nx_ = 0;

    /**
     * @brief Number of cells in the y-direction
     * 
     */
    int ny_ = 0;

    /**
     * @brief X-lower coordinate
     * 
     */
    double x_lower_ = 0;

    /**
     * @brief X-upper coordinate
     * 
     */
    double x_upper_ = 0;
    
    /**
     * @brief Y-lower coordinate
     * 
     */
    double y_lower_ = 0;

    /**
     * @brief Y-upper coordinate
     * 
     */
    double y_upper_ = 0;

    /**
     * @brief Grid spacing in the x-direction
     * 
     */
    double dx_ = 0;

    /**
     * @brief Grid spacing in the y-direction
     * 
     */
    double dy_ = 0;

public:

    // Petsc::DataManagement dm = PETSC_NULLPTR;

    /**
     * @brief Construct a new FiniteVolumeGrid object (default)
     * 
     */
    FiniteVolumeGrid();

    /**
     * @brief Construct a new FiniteVolumeGrid object
     * 
     * @param comm MPI communicator
     * @param nx Number of cells in the x-direction
     * @param x_lower Lower x-coordinate
     * @param x_upper Upper x-coordinate
     * @param ny Number of cells in the y-direction
     * @param y_lower Lower y-coordinate
     * @param y_upper Upper y-coordinate
     */
    FiniteVolumeGrid(MPI::Communicator comm, int nx, double x_lower, double x_upper, int ny, double y_lower, double y_upper);

    /**
     * @brief Destroy the FiniteVolumeGrid object
     * 
     */
    ~FiniteVolumeGrid();

    /**
     * @brief Creates the PETSc components of the grid
     * 
     * @return Petsc::ErrorCode 
     */
    Petsc::ErrorCode create();

    /**
     * @brief Sets the PETSc components from options of the grid
     * 
     * @return Petsc::ErrorCode 
     */
    Petsc::ErrorCode setFromOptions();

    /**
     * @brief Sets the grid up
     * 
     * @return Petsc::ErrorCode 
     */
    Petsc::ErrorCode setup();

    /**
     * @brief Returns the coordinate of the cell center at `index` for dimension `dim`
     * 
     * @param dim Dimension to query
     * @param index Index of cell
     * @return double 
     */
    double point(DimensionIndex dim, int index);

    /**
     * @brief Name of grid
     * 
     * @return std::string 
     */
    std::string name();

    /**
     * @brief Get the number of cells in the x-direction
     * 
     * @return std::size_t 
     */
    std::size_t nx();

    /**
     * @brief Get the number of cells in the y-direction
     * 
     * @return std::size_t 
     */
    std::size_t ny();

    /**
     * @brief Get the lower x-coordinate
     * 
     * @return double 
     */
    double xLower();

    /**
     * @brief Get the upper x-coordinate
     * 
     * @return double 
     */
    double xUpper();

    /**
     * @brief Get the lower y-coordinate
     * 
     * @return double 
     */
    double yLower();

    /**
     * @brief Get the upper y-coordinate
     * 
     * @return double 
     */
    double yUpper();

    /**
     * @brief Get the x-grid spacing
     * 
     * @return double 
     */
    double dx();

    /**
     * @brief Get the y-grid spacing
     * 
     * @return double 
     */
    double dy();

    /**
     * @brief Returns the coordinate of the cell center at `index` for dimension `DIM`
     * 
     * @sa `point`
     * 
     * @param DIM Dimension to query
     * @param index Index of cell
     * @return double 
     */
    double operator()(std::size_t DIM, std::size_t index);

};

namespace MPI {

/**
 * @brief Function overload of @sa `broadcast` for EllipticForest::FiniteVolumeGrid
 * 
 * @param grid Grid to communicate
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<>
int broadcast(FiniteVolumeGrid& grid, int root, MPI::Communicator comm);

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_GRID_HPP_