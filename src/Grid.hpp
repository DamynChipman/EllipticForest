#ifndef GRID_HPP_
#define GRID_HPP_

#include <string>
#include <petsc.h>
#include <petscdmda.h>

#include "MPI.hpp"

namespace EllipticForest {

enum DimensionIndex {
    X,
    Y
};

namespace Petsc {
    using DataManagement = DM;
    using ErrorCode = PetscErrorCode;
} // NAMESPACE : Petsc

class FiniteVolumeGrid : public MPI::MPIObject {

protected:

    int nx;
    int ny;
    double x_lower;
    double x_upper;
    double y_lower;
    double y_upper;
    double dx;
    double dy;
    Petsc::DataManagement dmda;

public:

    FiniteVolumeGrid();
    
    FiniteVolumeGrid(MPI::Communicator comm, int nx, int ny, double x_lower, double x_upper, double y_lower, double y_upper);

    Petsc::ErrorCode create();

    double point(DimensionIndex dim, int index);

}

} // NAMESPACE : EllipticForest

#endif GRID_HPP_