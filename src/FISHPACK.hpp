#ifndef FISHPACK_HPP_
#define FISHPACK_HPP_

#include "PatchGrid.hpp"
#include "PatchSolver.hpp"
#include "Patch.hpp"
#include "Quadtree.hpp"
#include "HPSAlgorithm.hpp"

namespace EllipticForest {

namespace FISHPACK {

// ---=====================---
// FISHPACK Finite Volume Grid
// ---=====================---

class FISHPACKFVGrid : public PatchGridBase<double> {

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

public:

    FISHPACKFVGrid();
    FISHPACKFVGrid(std::size_t nPointsX, std::size_t nPointsY, double xLower, double xUpper, double yLower, double yUpper);

    std::string name() { return name_; }
    std::size_t nPointsX() { return nPointsX_; }
    std::size_t nPointsY() { return nPointsY_; }
    double xLower() { return xLower_; }
    double xUpper() { return xUpper_; }
    double yLower() { return yLower_; }
    double yUpper() { return yUpper_; }
    double dx() { return dx_; }
    double dy() { return dy_; }

    double operator()(std::size_t DIM, std::size_t index);

};

// ---=============================---
// FISHPACK Finite Volume Patch Solver
// ---=============================---

extern "C" {
	void hstcrt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR);
	void hstcrtt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR, double* W);
}

class FISHPACKFVSolver : public PatchSolverBase<double> {

    FISHPACKFVSolver() {}

    std::string name();
    Vector<double> solve(FISHPACKFVGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    Vector<double> mapD2N(FISHPACKFVGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    Matrix<double> buildD2N(FISHPACKFVGrid& grid);

};

// ---======================---
// FISHPACK Finite Volume Patch
// ---======================---

struct FISHPACKPatch : public PatchBase<double> {

    // Metadata
	int ID = -1;									    // Patch's global ID
	int level = -1;								    	// Level in tree
	bool isLeaf = false;					    		// Flag for if patch is a leaf
	int nCellsLeaf = -1; 					   			// Storage for number of cells on leaf patch side
	Vector<int> nPatchSideVector = {0, 0, 0, 0};	    // To keep track of patch's side based on children

	// Patch grid information
	FISHPACKFVGrid grid;    		  	// Grid information

	// Data matrices
	Matrix<double> T{};						// DtN Matrix
	Matrix<double> T_prime{};	    		// Horizontal Merge DtN Matrix
	Matrix<double> S{};						// Solution Matrix
	Matrix<double> S_prime{};		    	// Horizontal Merge Solution Matrix
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
	Vector<double> w_prime{};		    	// Horizontal Particular Solution Vector

    // Pointers to finer and coarser versions of itself
    FISHPACKPatch* finer = nullptr;          // Finer version of itself
    FISHPACKPatch* coarser = nullptr;        // Coarser version of itself
    bool hasFiner = false;                              // Flag for if patch has finer version of itself
    bool hasCoarsened = false;                          // Flag for it patch has coarser version of itself

    FISHPACKPatch() {}
    // FISHPACKPatch(FISHPACKFVGrid& grid, int ID, int level, bool isLeaf);

    FISHPACKPatch& operator=(const FISHPACKPatch& rhs);

};

// ---=========================---
// FISHPACK Finite Volume Quadtree
// ---=========================---

class FISHPACKQuadtree : public Quadtree<FISHPACKPatch> {

public:

    FISHPACKQuadtree();
    FISHPACKQuadtree(p4est_t* p4est);

    FISHPACKPatch initData(FISHPACKPatch& parentData, std::size_t level, std::size_t index);

};

// ---===========================---
// FISHPACK Finite Volume HPS Method
// ---===========================---

class FISHPACKHPSMethod : public HPSAlgorithmBase<FISHPACKPatch> {

public:

    FISHPACKHPSMethod(FISHPACKPatch rootPatch, p4est_t* p4est);

protected:

    virtual void setupStage();
    virtual void merge4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    virtual void upwards4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    virtual void split1to4(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);

private:

    p4est_t* p4est_;
    FISHPACKPatch rootPatch_;

};

} // NAMESPACE : FISHPACK

} // NAMESPACE : EllipticForest

#endif // FISHPACK_HPP_