#ifndef PETSC_HPP_
#define PETSC_HPP_

#include <petsc.h>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "PatchGrid.hpp"
#include "PatchSolver.hpp"
#include "Patch.hpp"
#include "QuadNode.hpp"
#include "MPI.hpp"

namespace EllipticForest {

namespace Petsc {

#define DTN_OPTIMIZE 1

// class PetscVector : public Vector<double> {

// protected:

//     Vec vec_;

// public:

//     PetscVector();
//     PetscVector(MPI_Comm comm, int size);
//     PetscVector(MPI_Comm comm, Vector<double>& vector);




// };

// class PetscMatrix : public Matrix<double> {

// protected:

//     Mat mat_;

// };

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

    DM& dm() {return dm_; }

    virtual std::string name();
    virtual std::size_t nx();
    virtual std::size_t ny();
    virtual double xLower();
    virtual double xUpper();
    virtual double yLower();
    virtual double yUpper();
    virtual double dx();
    virtual double dy();
    virtual double operator()(std::size_t DIM, std::size_t index);

};

class PetscPatchSolver : public MPI::MPIObject {

protected:

    KSP ksp_;
    DM da_;
    Mat A_;

public:

    PetscPatchSolver();
    PetscPatchSolver(MPI_Comm comm);

    virtual std::string name();
    virtual Vector<double> solve(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    virtual Vector<double> mapD2N(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    virtual Matrix<double> buildD2N(PetscGrid& grid);
    virtual Vector<double> particularNeumannData(PetscGrid& grid, Vector<double>& rhsData);

    void setAlphaFunction(std::function<double(double, double)> fn);
    void setBetaFunction(std::function<double(double, double)> fn);
    void setLambdaFunction(std::function<double(double, double)> fn);
    void setRHSFunction(std::function<double(double, double)> fn);

    // Vector<double> computeDiffusionVector(PetscGrid& grid, std::function<double(double x, double y)> diffusionFunction);
    // Vector<double> computeLambdaVector(PetscGrid& grid, std::function<double(double x, double y)> lambdaFunction);
    // Vector<double> computeLoadVector(PetscGrid& grid, std::function<double(double x, double y)> rhsFunction);

    std::function<double(double x, double y)> alphaFunction;
    std::function<double(double x, double y)> betaFunction;
    std::function<double(double x, double y)> lambdaFunction;
    std::function<double(double x, double y)> rhsFunction;
    int gridIndex2MatrixIndex(int i, int j, int nx, int ny);

};

class PetscPatch : public PatchBase<PetscPatch, PetscGrid, PetscPatchSolver, double> {

public:

    PetscPatch();
    PetscPatch(PetscGrid grid);

    virtual std::string name();
    virtual PetscGrid& grid();
    virtual PetscPatch buildChild(std::size_t childIndex);
    // virtual PetscMatrix& matrixX();
    // virtual PetscMatrix& matrixH();
    // virtual PetscMatrix& matrixS();
    // virtual PetscMatrix& matrixT();
    // virtual PetscVector& vectorU();
    // virtual PetscVector& vectorG();
    // virtual PetscVector& vectorV();
    // virtual PetscVector& vectorF();
    // virtual PetscVector& vectorH();
    // virtual PetscVector& vectorW();
    virtual Matrix<double>& matrixX();
    virtual Matrix<double>& matrixH();
    virtual Matrix<double>& matrixS();
    virtual Matrix<double>& matrixT();
    virtual Vector<double>& vectorU();
    virtual Vector<double>& vectorG();
    virtual Vector<double>& vectorV();
    virtual Vector<double>& vectorF();
    virtual Vector<double>& vectorH();
    virtual Vector<double>& vectorW();

    double dataSize();
    std::string str();

private:

    PetscGrid grid_;

    Matrix<double> X{}, H{}, S{}, T{};
    Vector<double> u{}, g{}, v{}, f{}, h{}, w{};

};

class PetscPatchNodeFactory : public AbstractNodeFactory<PetscPatch>, public ::EllipticForest::MPI::MPIObject {

public:

    PetscPatchNodeFactory();
    PetscPatchNodeFactory(MPI_Comm comm);

    Node<PetscPatch>* createNode(PetscPatch data, std::string path, int level, int pfirst, int plast);
    Node<PetscPatch>* createChildNode(Node<PetscPatch>* parentNode, int siblingID, int pfirst, int plast);
    Node<PetscPatch>* createParentNode(std::vector<Node<PetscPatch>*> childNodes, int pfirst, int plast);

};

} // NAMESPACE : Petsc

namespace MPI {

template<>
int broadcast(Petsc::PetscGrid& grid, int root, MPI_Comm comm);

template<>
int broadcast(Petsc::PetscPatch& patch, int root, MPI_Comm comm);

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest

#endif // PETSC_HPP_