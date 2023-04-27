#ifndef PETSC_HPP_
#define PETSC_HPP_

#include <petsc.h>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"
#include "PatchSolver.hpp"
#include "Patch.hpp"

namespace EllipticForest {

namespace Petsc {

class PetscVector : public Vector<double> {

protected:

    Vec vec_;

};

class PetscMatrix : public Matrix<double> {

protected:

    Mat mat_;

};

class PetscGrid : public PatchGridBase<double> {

protected:

    DM dm_;
    std::string name_ = "PETScGrid";
    std::size_t nx_;
    std::size_t ny_;
    double xLower_;
    double xUpper_;
    double yLower_;
    double yUpper_;
    double dx_;
    double dy_;

public:

    PetscGrid();
    PetscGrid(int nx, int ny, double xLower, double xUpper, double yLower, double yUpper);

    PetscErrorCode create();

    virtual std::string name();
    virtual std::size_t nPointsX();
    virtual std::size_t nPointsY();
    virtual double xLower();
    virtual double xUpper();
    virtual double yLower();
    virtual double yUpper();
    virtual double dx();
    virtual double dy();
    virtual double operator()(std::size_t DIM, std::size_t index);

};

class PetscPatchSolver : public PatchSolverBase<double> {

protected:

    KSP ksp_;

public:

    PetscPatchSolver();

    virtual std::string name();
    virtual PetscVector solve(PetscGrid& grid, PetscVector& dirichletData, PetscVector& rhsData);
    virtual PetscVector mapD2N(PetscGrid& grid, PetscVector& dirichletData, PetscVector& rhsData);
    virtual PetscMatrix buildD2N(PetscGrid& grid);

    PetscVector computeDiffusionVector(PetscGrid& grid, std::function<double(double x, double y)> diffusionFunction);
    PetscVector computeLambdaVector(PetscGrid& grid, std::function<double(double x, double y)> lambdaFunction);
    PetscVector computeLoadVector(PetscGrid& grid, std::function<double(double x, double y)> loadFunction);

};

class PetscPatch : public PatchBase<PetscPatch, PetscGrid, PetscPatchSolver, double> {

public:

    PetscPatch();
    PetscPatch(PetscGrid grid);

    virtual std::string name();
    virtual PetscGrid& grid();
    virtual PetscPatch buildChild(std::size_t childIndex);
    virtual PetscMatrix& matrixX();
    virtual PetscMatrix& matrixH();
    virtual PetscMatrix& matrixS();
    virtual PetscMatrix& matrixT();
    virtual PetscVector& vectorU();
    virtual PetscVector& vectorG();
    virtual PetscVector& vectorV();
    virtual PetscVector& vectorF();
    virtual PetscVector& vectorH();
    virtual PetscVector& vectorW();

private:

    PetscGrid grid_;

};

} // NAMESPACE : Petsc

} // NAMESPACE : EllipticForest

#endif // PETSC_HPP_