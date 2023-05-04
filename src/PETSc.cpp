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
    int dofVertex = 0;
    int dofSide = 0;
    int dofCell = 1;
    int stencilWidth = 1;
    ierr = DMStagCreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx_, ny_, PETSC_DECIDE, PETSC_DECIDE, dofVertex, dofSide, dofCell, DMSTAG_STENCIL_STAR, stencilWidth, nullptr, nullptr, &dm_); CHKERRQ(ierr);
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

Vector<double> PetscPatchSolver::solve(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // Unpack Dirichlet data
    int nSide = grid.nPointsX();
    Vector<double> gWest = dirichletData.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichletData.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichletData.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichletData.getSegment(3*nSide, nSide);

    // Get Petsc data
    DM& dm = grid.dm();

}

Vector<double> PetscPatchSolver::mapD2N(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {



}

Matrix<double> PetscPatchSolver::buildD2N(PetscGrid& grid) {



}

void PetscPatchSolver::setAlphaFunction(std::function<double(double, double)> fn) {
    alphaFunction = fn;
}

void PetscPatchSolver::setBetaFunction(std::function<double(double, double)> fn) {
    betaFunction = fn;
}

void PetscPatchSolver::setLambdaFunction(std::function<double(double, double)> fn) {
    lambdaFunction = fn;
}

void PetscPatchSolver::setLoadFunction(std::function<double(double, double)> fn) {
    loadFunction = fn;
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

double PetscPatch::dataSize() {
    return 0.0;
}

Matrix<double>& PetscPatch::matrixX() {
    return X;
}

Matrix<double>& PetscPatch::matrixH() {
    return H;
}

Matrix<double>& PetscPatch::matrixS() {
    return S;
}

Matrix<double>& PetscPatch::matrixT() {
    return T;
}

Vector<double>& PetscPatch::vectorU() {
    return u;
}

Vector<double>& PetscPatch::vectorG() {
    return g;
}

Vector<double>& PetscPatch::vectorV() {
    return v;
}

Vector<double>& PetscPatch::vectorF() {
    return f;
}

Vector<double>& PetscPatch::vectorH() {
    return h;
}

Vector<double>& PetscPatch::vectorW() {
    return w;
}



} // NAMESPACE : Petsc

} // NAMESPACE : EllipticForest