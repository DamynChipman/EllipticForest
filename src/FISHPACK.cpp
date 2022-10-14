#include "FISHPACK.hpp"

namespace EllipticForest {

namespace FISHPACK {

#define DTN_OPTIMIZE 1

// ---=====================---
// FISHPACK Finite Volume Grid
// ---=====================---

FISHPACKFVGrid::FISHPACKFVGrid() :
    nPointsX_(0),
    nPointsY_(0),
    xLower_(-1.0),
    xUpper_(1.0),
    yLower_(-1.0),
    yUpper_(1.0),
    dx_(0),
    dy_(0)
        {}

FISHPACKFVGrid::FISHPACKFVGrid(std::size_t nPointsX, std::size_t nPointsY, double xLower, double xUpper, double yLower, double yUpper) :
        nPointsX_(nPointsX),
        nPointsY_(nPointsY),
        xLower_(xLower),
        xUpper_(xUpper),
        yLower_(yLower),
        yUpper_(yUpper),
        dx_((xUpper - xLower) / nPointsX),
        dy_((yUpper - yLower) / nPointsY)
            {}

double FISHPACKFVGrid::operator()(std::size_t DIM, std::size_t index)  {
    if (DIM == XDIM) {
        if (index >= nPointsX_ || index < 0) {
            std::string errorMessage = "[EllipticForest::FISHPACKFVGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tnPointsX = " + std::to_string(nPointsX_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return (xLower_ + dx_/2) + index*dx_;
    }
    else if (DIM == YDIM) {
        if (index >= nPointsY_ || index < 0) {
            std::string errorMessage = "[EllipticForest::FISHPACKFVGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tnPointsY = " + std::to_string(nPointsY_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return (yLower_ + dy_/2) + index*dy_;
    }
    else {
        std::string errorMessage = "[EllipticForest::FISHPACKFVGrid::operator()] `DIM` is not a correct index:\n";
        errorMessage += "DIM = " + std::to_string(DIM) + "\n";
        std::cerr << errorMessage << std::endl;
        throw std::invalid_argument(errorMessage);
    }
}

// ---=============================---
// FISHPACK Finite Volume Patch Solver
// ---=============================---

std::string FISHPACKFVSolver::name() {
    return "FISHPACK90Solver";
}

Vector<double> FISHPACKFVSolver::solve(FISHPACKFVGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // Unpack Dirichlet Data
	int nSide = grid.nPointsX();
	Vector<double> gWest = dirichletData.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichletData.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichletData.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichletData.getSegment(3*nSide, nSide);

	// Transpose RHS data for FORTRAN call
	Vector<double> fT(nSide * nSide);
	for (int i = 0; i < grid.nPointsX(); i++) {
		for (int j = 0; j < grid.nPointsY(); j++) {
			fT[i + j*nSide] = rhsData[j + i*nSide];
		}
	}

	// Setup FORTRAN call to FISHPACK
	double A = grid.xLower();
	double B = grid.xUpper();
	int M = grid.nPointsX();
	int MBDCND = 1;
	double* BDA = gWest.dataPointer();
	double* BDB = gEast.dataPointer();
	double C = grid.yLower();
	double D = grid.yUpper();
	int N = grid.nPointsY();
	int NBDCND = 1;
	double* BDC = gSouth.dataPointer();
	double* BDD = gNorth.dataPointer();
	double ELMBDA = 0; // @TODO: Implement or get lambda value
	double* F = fT.dataPointer();
	int IDIMF = M;
	double PERTRB;
	int IERROR;
	int WSIZE = 13*M + 4*N + M*((int)log2(N));
	double* W = (double*) malloc(WSIZE*sizeof(double));

	// Make FORTRAN call to FISHPACK
	hstcrtt_(&A, &B, &M, &MBDCND, BDA, BDB,
			&C, &D, &N, &NBDCND, BDC, BDD,
			&ELMBDA, F, &IDIMF, &PERTRB, &IERROR, W);
	if (IERROR != 1) {
		std::cerr << "[fc2d_hps_FISHPACK_solver::solve] WARNING: call to hstcrt_ returned non-zero error value: IERROR = " << IERROR << std::endl;
	}

	// Move FISHPACK solution into Vector for output
	Vector<double> solution(grid.nPointsX() * grid.nPointsY());
	for (int i = 0; i < grid.nPointsX(); i++) {
		for (int j = 0; j < grid.nPointsY(); j++) {
			solution[j + i*nSide] = F[i + j*nSide];
		}
	}

	return solution; // return rhsData;

}

Vector<double> FISHPACKFVSolver::mapD2N(FISHPACKFVGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // Unpack grid data
	int nSide = grid.nPointsX();

	// Unpack Dirichlet data
	Vector<double> gWest = dirichletData.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichletData.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichletData.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichletData.getSegment(3*nSide, nSide);

	// Compute solution on interior nodes
	Vector<double> u = solve(grid, dirichletData, rhsData);

	// Get interior edge cell data and compute Neumann data
	//    Interior cell data
	Vector<double> uWest(nSide);
	Vector<double> uEast(nSide);
	Vector<double> uSouth(nSide);
	Vector<double> uNorth(nSide);
	
	//    Fill interior cell data
	for (int j = 0; j < nSide; j++) {
		uWest[j] = u[j];
		uEast[j] = u[(nSide-1)*nSide + j];
	}
	for (int i = 0; i < nSide; i++) {
		uSouth[i] = u[i*nSide];
		uNorth[i] = u[(i+1)*nSide - 1];
	}

	//    Neumann data
	double dtn_x = 2.0 / grid.dx();
	double dtn_y = 2.0 / grid.dy();
	Vector<double> hWest(nSide);
	Vector<double> hEast(nSide);
	Vector<double> hSouth(nSide);
	Vector<double> hNorth(nSide);
	for (int i = 0; i < nSide; i++) {
		hWest[i]  = (dtn_x)  * (uWest[i] - gWest[i]);
		hEast[i]  = (-dtn_x) * (uEast[i] - gEast[i]);
		hSouth[i] = (dtn_y)  * (uSouth[i] - gSouth[i]);
		hNorth[i] = (-dtn_y) * (uNorth[i] - gNorth[i]);
	}

	//    Column stack and return
	Vector<double> neumannData(4*nSide);
	neumannData.setSegment(0*nSide, hWest);
	neumannData.setSegment(1*nSide, hEast);
	neumannData.setSegment(2*nSide, hSouth);
	neumannData.setSegment(3*nSide, hNorth);
	return neumannData;

}

Matrix<double> FISHPACKFVSolver::buildD2N(FISHPACKFVGrid& grid) {

    std::size_t N = grid.nPointsX();
	std::size_t M = 4*N;
	Matrix<double> T(M, M);
	Vector<double> e_hat_j(M, 0.0);
	Vector<double> f_zero(N*N, 0.0);
	Vector<double> col_j(M);

#if DTN_OPTIMIZE
	// Iterate through first side of grid to form T
	// Compute first column of block T
	for (int j = 0; j < N; j++) {
		e_hat_j[j] = 1.0;
		col_j = mapD2N(grid, e_hat_j, f_zero);
		T.setColumn(j, col_j);
		e_hat_j[j] = 0.0;
	}

	// Extract blocks of T
	Matrix<double> T_WW = T.getBlock(0*N, 0*N, N, N);
	Matrix<double> T_EW = T.getBlock(1*N, 0*N, N, N);
	Matrix<double> T_SW = T.getBlock(2*N, 0*N, N, N);
	Matrix<double> T_NW = T.getBlock(3*N, 0*N, N, N);

	// Define other blocks in terms of first block column
	// T_WE = -T_EW
	// T_EE = -T_WW
	// T_SE = -T_NW^T
	// T_NE = Reversed columns from T_NW
	// 
	// T_WS = T_SW
	// T_ES = T_NW
	// T_SS = T_WW
	// T_NS = T_EW
	//
	// T_WN = -T_NW
	// T_EN = T_NE
	// T_SN = -T_EW
	// T_NN = -T_WW
	Matrix<double> T_WE(N, N);
	Matrix<double> T_EE(N, N);
	Matrix<double> T_SE(N, N);
	Matrix<double> T_NE(N, N);

	Matrix<double> T_WS(N, N);
	Matrix<double> T_ES(N, N);
	Matrix<double> T_SS(N, N);
	Matrix<double> T_NS(N, N);

	Matrix<double> T_WN(N, N);
	Matrix<double> T_EN(N, N);
	Matrix<double> T_SN(N, N);
	Matrix<double> T_NN(N, N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			T_WE(i,j) = -T_EW(i,j);
			T_EE(i,j) = -T_WW(i,j);
			T_SE(i,j) = -T_NW(j,i);
			T_NE(i,j) = T_NW((N-1) - i, j);

			T_WS(i,j) = T_SW(i,j);
			T_ES(i,j) = T_NW(i,j);
			T_SS(i,j) = T_WW(i,j);
			T_NS(i,j) = T_EW(i,j);

			T_WN(i,j) = -T_NW(j,i);
			T_EN(i,j) = T_NE(i,j);
			T_SN(i,j) = -T_EW(i,j);
			T_NN(i,j) = -T_WW(i,j);
		}
	}

	// Set blocks into T
	T.setBlock(0*N, 1*N, T_WE);
	T.setBlock(1*N, 1*N, T_EE);
	T.setBlock(2*N, 1*N, T_SE);
	T.setBlock(3*N, 1*N, T_NE);
	
	T.setBlock(0*N, 2*N, T_WS);
	T.setBlock(1*N, 2*N, T_ES);
	T.setBlock(2*N, 2*N, T_SS);
	T.setBlock(3*N, 2*N, T_NS);

	T.setBlock(0*N, 3*N, T_WN);
	T.setBlock(1*N, 3*N, T_EN);
	T.setBlock(2*N, 3*N, T_SN);
	T.setBlock(3*N, 3*N, T_NN);
#else
	// Iterate through all points on boundary to form T
	for (int j = 0; j < M; j++) {
		e_hat_j[j] = 1.0;
		col_j = this->dtn(grid, e_hat_j, f_zero);
		T.setColumn(j, col_j);
		e_hat_j[j] = 0.0;
	}
#endif
	return T;

}

// ---======================---
// FISHPACK Finite Volume Patch
// ---======================---

// FISHPACKPatch::FISHPACKPatch(FISHPACKFVGrid grid, int ID, int level, bool isLeaf) :
//     grid(grid),
//     ID(ID),
//     level(level),
//     isLeaf(isLeaf) {

//     if (isLeaf) {
//         nPatchSideVector = {1, 1, 1, 1};
//     }

// }

FISHPACKPatch& FISHPACKPatch::operator=(const FISHPACKPatch& rhs) {

    if (&rhs != this) {
        ID = rhs.ID;
        level = rhs.level;
        isLeaf = rhs.isLeaf;
        nCellsLeaf = rhs.nCellsLeaf;
        nPatchSideVector = rhs.nPatchSideVector;
        grid = rhs.grid;
        T = rhs.T;
        T_prime = rhs.T_prime;
        S = rhs.S;
        S_prime = rhs.S_prime;
        X = rhs.X;
        u = rhs.u;
        g = rhs.g;
        v = rhs.v;
        f = rhs.f;
        h = rhs.h;
        w = rhs.w;
        w_prime = rhs.w_prime;
        finer = rhs.finer;
        coarser = rhs.coarser;
        hasFiner = rhs.hasFiner;
        hasCoarsened = rhs.hasCoarsened;
    }
    return *this;

}

// ---=========================---
// FISHPACK Finite Volume Quadtree
// ---=========================---

FISHPACKQuadtree::FISHPACKQuadtree() {}

FISHPACKQuadtree::FISHPACKQuadtree(p4est_t* p4est) :
    Quadtree<FISHPACKPatch>(p4est)
        {}

FISHPACKPatch FISHPACKQuadtree::initData(FISHPACKPatch& parentData, std::size_t level, std::size_t index) {
    
    // parentData.isLeaf = false;
    FISHPACKFVGrid& parentGrid = parentData.grid;

    std::size_t nx = parentGrid.nPointsX();
    std::size_t ny = parentGrid.nPointsY();
    double xMid = (parentGrid.xLower() + parentGrid.xUpper()) / 2.0;
    double yMid = (parentGrid.yLower() + parentGrid.yUpper()) / 2.0;
    double xLower, xUpper, yLower, yUpper;
    if (index % 4 == 0) {
        // Lower left
        xLower = parentGrid.xLower();
        xUpper = xMid;
        yLower = parentGrid.yLower();
        yUpper = yMid;
    }
    else if (index % 4 == 1) {
        // Lower right
        xLower = xMid;
        xUpper = parentGrid.xUpper();
        yLower = parentGrid.yLower();
        yUpper = yMid;
    }
    else if (index % 4 == 2) {
        // Upper left
        xLower = parentGrid.xLower();
        xUpper = xMid;
        yLower = yMid;
        yUpper = parentGrid.yUpper();
    }
    else if (index % 4 == 3) {
        // Upper right
        xLower = xMid;
        xUpper = parentGrid.xUpper();
        yLower = yMid;
        yUpper = parentGrid.yUpper();
    }
    FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);

    FISHPACKPatch patch;
    patch.grid = grid;
    patch.level = level;
    if (this->childIndices_[level][index] == -1) {
        patch.isLeaf = true;
    }
    else {
        patch.isLeaf = false;
    }
    // patch.isLeaf = true;
    patch.nCellsLeaf = nx;

    return patch;

}

// ---===========================---
// FISHPACK Finite Volume HPS Method
// ---===========================---

FISHPACKHPSMethod::FISHPACKHPSMethod(FISHPACKPatch rootPatch, p4est_t* p4est) :
    p4est_(p4est),
    rootPatch_(rootPatch)
        {}

void FISHPACKHPSMethod::setupStage() {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Begin FISHPACK-HPS Setup Stage");

    // Initialize quadtree
    this->quadtree = new FISHPACKQuadtree(p4est_);
    quadtree->build(rootPatch_);

    int currentID = 0;
    quadtree->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) patch.ID = currentID++;
        else patch.ID = -1;
    });

    // quadtree->traversePreOrder([](FISHPACKPatch& patch){
    //     std::cout << "patch.ID = " << patch.ID << std::endl;
    // });

    app.log("End FISHPACK-HPS Setup Stage");

}

void FISHPACKHPSMethod::merge4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Merging:");
    app.log("alpha = %i", alpha.ID);
    app.log("beta = %i", beta.ID);
    app.log("gamma = %i", gamma.ID);
    app.log("omega = %i", omega.ID);

}

void FISHPACKHPSMethod::upwards4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Upwards:");
    app.log("alpha = %i", alpha.ID);
    app.log("beta = %i", beta.ID);
    app.log("gamma = %i", gamma.ID);
    app.log("omega = %i", omega.ID);

}

void FISHPACKHPSMethod::split1to4(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Splitting:");
    app.log("alpha = %i", alpha.ID);
    app.log("beta = %i", beta.ID);
    app.log("gamma = %i", gamma.ID);
    app.log("omega = %i", omega.ID);

}


} // NAMESPACE : FISHPACK

} // NAMESPACE : EllipticSolver