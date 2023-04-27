#include "PETSc.hpp"

namespace EllipticForest {

namespace Petsc {

PetscGrid::PetscGrid() {}

PetscGrid::PetscGrid(int nx, int ny, double xLower, double xUpper, double yLower, double yUpper) :
    nx_(nx), ny_(ny), xLower_(xLower), xUpper_(xUpper), yLower_(yLower), yUpper_(yUpper) {

    dx_ = (xUpper_ - xLower_) / (nx_);
    dy_ = (yUpper_ - xLower_) / (ny_);

}

PetscErrorCode PetscGrid::create() {

    PetscErrorCode ierr;
    int dofs = 1;
    int stencilWidth = 1;
    ierr = DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, nx_, ny_, PETSC_DECIDE, PETSC_DECIDE, dofs, stencilWidth, nullptr, nullptr, &dm_); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(dm_, xLower_, xUpper_, yLower_, yUpper_, 0, 0);
    ierr = DMSetFromOptions(dm_); CHKERRQ(ierr);
    ierr = DMSetUp(dm_); CHKERRQ(ierr);
    return ierr;

}

std::string PetscGrid::name() { return name_; }

std::size_t PetscGrid::nPointsX() { return nx_; }

std::size_t PetscGrid::nPointsY() { return ny_; }

double PetscGrid::xLower() { return xLower_; }

double PetscGrid::xUpper() { return xUpper_; }

double PetscGrid::yLower() { return yLower_; }

double PetscGrid::yUpper() { return yUpper_; }

double PetscGrid::dx() { return dx_; }

double PetscGrid::dy() { return dx_; }

double PetscGrid::operator()(std::size_t DIM, std::size_t index) {
    if (DIM == XDIM) {
        if (index >= nx_ || index < 0) {
            std::string errorMessage = "[EllipticForest::Petsc::PetscGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tnx = " + std::to_string(nx_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return (xLower_ + dx_/2) + index*dx_;
    }
    else if (DIM == YDIM) {
        if (index >= ny_ || index < 0) {
            std::string errorMessage = "[EllipticForest::Petsc::PetscGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tny = " + std::to_string(ny_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return (yLower_ + dy_/2) + index*dy_;
    }
    else {
        std::string errorMessage = "[EllipticForest::Petsc::PetscGrid::operator()] `DIM` is not a correct index:\n";
        errorMessage += "DIM = " + std::to_string(DIM) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
}

PetscPatchSolver::PetscPatchSolver() {}

std::string PetscPatchSolver::name() { return "PETScPatchSolver"; }

PetscVector PetscPatchSolver::solve(PetscGrid& grid, PetscVector& dirichletData, PetscVector& rhsData) {



}

PetscVector PetscPatchSolver::mapD2N(PetscGrid& grid, PetscVector& dirichletData, PetscVector& rhsData) {



}

PetscMatrix PetscPatchSolver::buildD2N(PetscGrid& grid) {



}

PetscPatch::PetscPatch() {}

PetscPatch::PetscPatch(PetscGrid grid) :
    grid_(grid)
        {}

std::string PetscPatch::name() {

}

PetscGrid& PetscPatch::grid() {

}

PetscPatch PetscPatch::buildChild(std::size_t childIndex) {

}

PetscMatrix& PetscPatch::matrixX() {

}

PetscMatrix& PetscPatch::matrixH() {

}

PetscMatrix& PetscPatch::matrixS() {

}

PetscMatrix& PetscPatch::matrixT() {

}

PetscVector& PetscPatch::vectorU() {

}

PetscVector& PetscPatch::vectorG() {

}

PetscVector& PetscPatch::vectorV() {

}

PetscVector& PetscPatch::vectorF() {

}

PetscVector& PetscPatch::vectorH() {

}

PetscVector& PetscPatch::vectorW() {

}



} // NAMESPACE : Petsc

} // NAMESPACE : EllipticForest