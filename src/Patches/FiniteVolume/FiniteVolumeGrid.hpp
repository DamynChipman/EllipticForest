#ifndef FINITE_VOLUME_GRID_HPP_
#define FINITE_VOLUME_GRID_HPP_

#include <string>

#include <petsc.h>
#include <petscdmda.h>

#include "../../EllipticForestApp.hpp"
#include "../../MPI.hpp"
#include "../../PatchGrid.hpp"

namespace EllipticForest {

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

    int nx_ = 0, ny_ = 0;
    double x_lower_ = 0, x_upper_ = 0, y_lower_ = 0, y_upper_ = 0, dx_ = 0, dy_ = 0;

public:

    Petsc::DataManagement dm = PETSC_NULLPTR;
    
    FiniteVolumeGrid();

    FiniteVolumeGrid(MPI::Communicator comm, int nx, double x_lower, double x_upper, int ny, double y_lower, double y_upper);

    // // Copy constructor
    // FiniteVolumeGrid(FiniteVolumeGrid& copy_grid);

    // // Move constructor
    // FiniteVolumeGrid(FiniteVolumeGrid&& move_grid);

    // FiniteVolumeGrid& operator=(FiniteVolumeGrid& other);

    // FiniteVolumeGrid& operator=(FiniteVolumeGrid&& other);

    ~FiniteVolumeGrid();

    Petsc::ErrorCode create();
    Petsc::ErrorCode setFromOptions();
    Petsc::ErrorCode setup();

    double point(DimensionIndex dim, int index);

    std::string name();
    std::size_t nx();
    std::size_t ny();
    double xLower();
    double xUpper();
    double yLower();
    double yUpper();
    double dx();
    double dy();
    double operator()(std::size_t DIM, std::size_t index);

};

namespace MPI {

template<>
int broadcast(FiniteVolumeGrid& grid, int root, MPI::Communicator comm);

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_GRID_HPP_