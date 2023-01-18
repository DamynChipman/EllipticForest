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

#if USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

namespace EllipticForest {

namespace FISHPACK {

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

    FISHPACKFVSolver() {}

    std::string name();
    Vector<double> solve(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    Vector<double> mapD2N(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData);
    Matrix<double> buildD2N(PatchGridBase<double>& grid);

};

// ---======================---
// FISHPACK Finite Volume Patch
// ---======================---

struct FISHPACKPatch : public PatchBase<double> {

    // Metadata
    int leafID = -1;                                    // Leaf level ID (p4est ID)
    int globalID = -1;                                  // Global ID (Quadtree ID)
	int level = -1;								    	// Level in tree
	bool isLeaf = false;					    		// Flag for if patch is a leaf
	// int nCellsLeaf = -1; 					   			// Storage for number of cells on leaf patch side // TODO: Remove and put in options
	// Vector<int> nPatchSideVector = {0, 0, 0, 0};	    // To keep track of patch's side based on children // TODO: Change to single value; rename nLeafSides
    // int nLeafSides = -1;
    // int nChildren = 0;
    int nCoarsens = 0;
    // int resolutionFactor;

	// Patch grid information
	FISHPACKFVGrid grid;    		  	// Grid information

	// Data matrices
	Matrix<double> T{};						// DtN Matrix
    Matrix<double> H{};
    // Matrix<double> Tprime{};                // Coarser version of T
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

    // Pointers to finer and coarser versions of itself
    // std::list<FISHPACKPatch*> versions;
    // FISHPACKPatch* finer = nullptr;          // Finer version of itself
    // FISHPACKPatch* coarser = nullptr;        // Coarser version of itself
    // bool hasFiner = false;                              // Flag for if patch has finer version of itself
    // bool hasCoarsened = false;                          // Flag for it patch has coarser version of itself
    // FISHPACKPatch* root;

    FISHPACKPatch();
    // FISHPACKPatch(FISHPACKFVGrid& grid, int ID, int level, bool isLeaf);

    // FISHPACKPatch& operator=(const FISHPACKPatch& rhs);
    std::string str();
    // void coarsen();
    // void coarsenUpwards();
    // void uncoarsen();
    // Matrix<double>& getT();
    // Vector<double>& getH();
    // Vector<double>& getG();
    // Vector<double>& getW();

};

// ---==============---
// FISHPACK PDE Problem
// ---==============---

class FISHPACKProblem : public EllipticProblemBase<double> {

public:

    FISHPACKProblem() {}
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

// ---=========================---
// FISHPACK Finite Volume Quadtree
// ---=========================---

class FISHPACKQuadtree : public Quadtree<FISHPACKPatch> {

public:

    FISHPACKQuadtree();
    FISHPACKQuadtree(p4est_t* p4est);

    FISHPACKPatch initData(FISHPACKPatch& parentData, std::size_t level, std::size_t index);

    void toVTK(std::string filename);

};

// ---===========================---
// FISHPACK Finite Volume HPS Method
// ---===========================---

class FISHPACKHPSMethod : public HPSAlgorithmBase<FISHPACKPatch, double> {

public:

    FISHPACKHPSMethod(FISHPACKProblem& PDE, FISHPACKPatch& rootPatch, p4est_t* p4est);
    void toVTK(std::string filename);
    virtual void setupStage();
    virtual void preSolveHook();

protected:

    virtual void merge4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    virtual void upwards4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    virtual void split1to4(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    virtual void setParticularData(FISHPACKPatch& patch);
    virtual void leafSolve(FISHPACKPatch& patch);

private:

    FISHPACKProblem& pde_;
    FISHPACKPatch& rootPatch_;
    p4est_t* p4est_;

    // Index sets
    Vector<int> IS_alpha_beta_;
    Vector<int> IS_alpha_gamma_;
    Vector<int> IS_alpha_omega_;
    Vector<int> IS_alpha_tau_;

    Vector<int> IS_beta_alpha_;
    Vector<int> IS_beta_gamma_;
    Vector<int> IS_beta_omega_;
    Vector<int> IS_beta_tau_;

    Vector<int> IS_gamma_alpha_;
    Vector<int> IS_gamma_beta_;
    Vector<int> IS_gamma_omega_;
    Vector<int> IS_gamma_tau_;

    Vector<int> IS_omega_alpha_;
    Vector<int> IS_omega_beta_;
    Vector<int> IS_omega_gamma_;
    Vector<int> IS_omega_tau_;

    // Blocks for X_tau
    Matrix<double> T_ag_ag;
    Matrix<double> T_ga_ga;
    Matrix<double> T_ag_ab;
    Matrix<double> T_ga_go;
    Matrix<double> T_bo_bo;
    Matrix<double> T_ob_ob;
    Matrix<double> T_bo_ba;
    Matrix<double> T_ob_og;
    Matrix<double> T_ab_ag;
    Matrix<double> T_ba_bo;
    Matrix<double> T_ab_ab;
    Matrix<double> T_ba_ba;
    Matrix<double> T_go_ga;
    Matrix<double> T_og_ob;
    Matrix<double> T_go_go;
    Matrix<double> T_og_og;

    // Blocks for S_tau
    Matrix<double> T_ag_at;
    Matrix<double> T_ga_gt;
    Matrix<double> T_bo_bt;
    Matrix<double> T_ob_ot;
    Matrix<double> T_ab_at;
    Matrix<double> T_ba_bt;
    Matrix<double> T_go_gt;
    Matrix<double> T_og_ot;

    // Blocks for T_tau
    Matrix<double> T_at_at;
    Matrix<double> T_bt_bt;
    Matrix<double> T_gt_gt;
    Matrix<double> T_ot_ot;
    Matrix<double> T_at_ag;
    Matrix<double> T_at_ab;
    Matrix<double> T_bt_bo;
    Matrix<double> T_bt_ba;
    Matrix<double> T_gt_ga;
    Matrix<double> T_gt_go;
    Matrix<double> T_ot_ob;
    Matrix<double> T_ot_og;

    // Blocks for particular problem (non-homogeneous)
    // Matrix<double> systemB;
    // Vector<double> hDiff;

    // Steps for the merge
    Vector<int> tagPatchesForCoarsening_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void coarsen_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void createIndexSets_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void createMatrixBlocks_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergeX_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergeS_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergeT_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void reorderOperators_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergePatch_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);

    // Steps for the upwards stage
    void coarsenUpwards_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergeW_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void mergeH_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void reorderOperatorsUpwards_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);

    // Steps for the split
    void uncoarsen_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);
    void applyS_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega);

};

} // NAMESPACE : FISHPACK

} // NAMESPACE : EllipticForest

#endif // FISHPACK_HPP_