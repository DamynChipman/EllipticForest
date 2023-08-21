#ifndef FISHPACK_HPP_
#define FISHPACK_HPP_

#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <list>

// #include <matplotlibcpp.h>
#include "PatchGrid.hpp"
#include "PatchSolver.hpp"
#include "Patch.hpp"
#include "EllipticProblem.hpp"
#include "Quadtree.hpp"
#include "HPSAlgorithm.hpp"
#include "SpecialMatrices.hpp"
#include "PlotUtils.hpp"
#include "VTK.hpp"
#include "MPI.hpp"

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

namespace EllipticForest {

namespace FISHPACK {

class FISHPACKProblem : public EllipticProblemBase<double> {

public:

    FISHPACKProblem() {}
    FISHPACKProblem(double lambda);
    void setU(std::function<double(double, double)> func) { u_ = func; }
    void setF(std::function<double(double, double)> func) { f_ = func; }
    void setDUDX(std::function<double(double, double)> func) { dudx_ = func; }
    void setDUDY(std::function<double(double, double)> func) { dudy_ = func; }

    double u(double x, double y) { return u_(x,y); }
    double f(double x, double y) { return f_(x,y); }
    double dudx(double x, double y) { return dudx_(x,y); }
    double dudy(double x, double y) { return dudy_(x,y); }

    virtual std::string name() { return "FISHPACKPDE"; }

protected:

    std::function<double(double, double)> u_;
    std::function<double(double, double)> f_;
    std::function<double(double, double)> dudx_;
    std::function<double(double, double)> dudy_;

};

// ---=====================---
// FISHPACK Finite Volume Grid
// ---=====================---

class FISHPACKFVGrid : public PatchGridBase<double>, public RectilinearGridNodeBase {

protected:

    std::string name_ = "FISHPACKFVGrid";
    std::size_t nPointsX_;
    std::size_t nPointsY_;
    double xLower_;
    double xUpper_;
    double yLower_;
    double yUpper_;
    double dx_;
    double dy_;
    Vector<double> xPoints_;
    Vector<double> yPoints_;

public:

    FISHPACKFVGrid();
    FISHPACKFVGrid(std::size_t nPointsX, std::size_t nPointsY, double xLower, double xUpper, double yLower, double yUpper);

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

    Vector<double>& xPoints();
    Vector<double>& yPoints();

    std::string getWholeExtent();
    std::string getExtent();

};

// ---=============================---
// FISHPACK Finite Volume Patch Solver
// ---=============================---

extern "C" {
	void hstcrt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR);
	void hstcrtt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR, double* W);
}

class FISHPACKFVSolver : public PatchSolverBase<double> {

public:

    // FISHPACKProblem pde;

    FISHPACKFVSolver();
    FISHPACKFVSolver(double lambda);

    virtual std::string name();
    virtual Vector<double> solve(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    virtual Vector<double> mapD2N(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    virtual Matrix<double> buildD2N(PatchGridBase<double>& grid);
    // virtual Vector<double> rhsData(PatchGridBase<double>& grid);

};

// ---======================---
// FISHPACK Finite Volume Patch
// ---======================---

class FISHPACKPatch : public PatchBase<FISHPACKPatch, FISHPACKFVGrid, FISHPACKFVSolver, double> {

public:

    // Metadata
    // int leafID = -1;                                    // Leaf level ID (p4est ID)
    // int globalID = -1;                                  // Global ID (Quadtree ID)
	// int level = -1;								    	// Level in tree
    // int nCoarsens = 0;
	// bool isLeaf = false;					    		// Flag for if patch is a leaf

    FISHPACKPatch();
    FISHPACKPatch(FISHPACKFVGrid grid);
    ~FISHPACKPatch();

    virtual std::string name();

	// Patch grid information
	// FISHPACKFVGrid grid;    		  	// Grid information
    virtual FISHPACKFVGrid& grid();

    virtual FISHPACKPatch buildChild(std::size_t childIndex);

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

    std::string str();
    double dataSize();

private:

    // Grid
    FISHPACKFVGrid grid_;

	// Data matrices
	Matrix<double> T{};						// DtN Matrix
    Matrix<double> H{};
	Matrix<double> S{};						// Solution Matrix
	Matrix<double> X{};						// Body Load Matrix
	
	// Solution data
	Vector<double> u{};						// Solution Vector
	Vector<double> g{};						// Dirichlet Vector
	Vector<double> v{};						// Neumann Vector

	// Homogeneous data
	Vector<double> f{};						// Poisson Vector

	// Particular data
	Vector<double> h{};						// Particular Neumann Vector
	Vector<double> w{};						// Particular Solution Vector

};

class FISHPACKPatchNodeFactory : public AbstractNodeFactory<FISHPACKPatch>, public ::EllipticForest::MPI::MPIObject {

public:

    FISHPACKPatchNodeFactory();
    FISHPACKPatchNodeFactory(MPI_Comm comm);

    Node<FISHPACKPatch>* createNode(FISHPACKPatch data, std::string path, int level, int pfirst, int plast);
    Node<FISHPACKPatch>* createChildNode(Node<FISHPACKPatch>* parentNode, int siblingID, int pfirst, int plast);
    Node<FISHPACKPatch>* createParentNode(std::vector<Node<FISHPACKPatch>*> childNodes, int pfirst, int plast);

};

} // NAMESPACE : FISHPACK

namespace MPI {

template<>
int broadcast(FISHPACK::FISHPACKFVGrid& grid, int root, MPI_Comm comm);

template<>
int broadcast(FISHPACK::FISHPACKPatch& patch, int root, MPI_Comm comm);

}

} // NAMESPACE : EllipticForest

#endif // FISHPACK_HPP_