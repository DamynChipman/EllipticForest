#include "FISHPACK.hpp"

namespace EllipticForest {

namespace FISHPACK {

#define DTN_OPTIMIZE 1

enum FISHPACK_BOUNDARY_SIDE {
    WEST,
    EAST,
    SOUTH,
    NORTH
};

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

Vector<double> FISHPACKFVSolver::solve(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

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
    // hstcrt_(&A, &B, &M, &MBDCND, BDA, BDB,
	// 		&C, &D, &N, &NBDCND, BDC, BDD,
	// 		&ELMBDA, F, &IDIMF, &PERTRB, &IERROR);
	// if (IERROR) {
	// 	std::cerr << "[EllipticForest::FISHPACK::FISHPACKFVSolver::solve] WARNING: call to hstcrt_ returned non-zero error value: IERROR = " << IERROR << std::endl;
	// }

	// Move FISHPACK solution into Vector for output
	Vector<double> solution(grid.nPointsX() * grid.nPointsY());
	for (int i = 0; i < grid.nPointsX(); i++) {
		for (int j = 0; j < grid.nPointsY(); j++) {
			solution[j + i*nSide] = F[i + j*nSide];
		}
	}

    free(W);

	return solution; // return rhsData;

}

Vector<double> FISHPACKFVSolver::mapD2N(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

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

Matrix<double> FISHPACKFVSolver::buildD2N(PatchGridBase<double>& grid) {

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

void FISHPACKPatch::coarsen() {

    // Copy metadata
    coarser = new FISHPACKPatch;
    coarser->ID = ID;
    coarser->level = level + 1;
    coarser->isLeaf = false;
	coarser->nCellsLeaf = nCellsLeaf;
	coarser->nPatchSideVector[WEST] = nPatchSideVector[WEST] / 2;
	coarser->nPatchSideVector[EAST] = nPatchSideVector[EAST] / 2;
	coarser->nPatchSideVector[SOUTH] = nPatchSideVector[SOUTH] / 2;
	coarser->nPatchSideVector[NORTH] = nPatchSideVector[NORTH] / 2;
	coarser->grid = FISHPACKFVGrid(grid.nPointsX()/2, grid.nPointsY()/2, grid.xLower(), grid.xUpper(), grid.yLower(), grid.yUpper());

    // Build L21
    int nFine = nCellsLeaf * nPatchSideVector[WEST];
    int nCoarse = nFine / 2;

    InterpolationMatrixFine2Coarse<double> L21Side(nCoarse);
    std::vector<Matrix<double>> L21Diagonals = {L21Side, L21Side, L21Side, L21Side};
    Matrix<double> L21Patch = blockDiagonalMatrix(L21Diagonals);

    // Build L12
    InterpolationMatrixCoarse2Fine<double> L12Side(nFine);
    std::vector<Matrix<double>> L12Diagonals = {L12Side, L12Side, L12Side, L12Side};
    Matrix<double> L12Patch = blockDiagonalMatrix(L12Diagonals);

    // D2N matrix
    // coarser->T = L21Patch * T;
    // coarser->T = coarser->T * L12Patch;

    // Solution matrix

    // Set flag
    hasCoarsened = true;
    coarser->hasCoarsened = false;

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

FISHPACKHPSMethod::FISHPACKHPSMethod(FISHPACKProblem PDE, FISHPACKPatch rootPatch, p4est_t* p4est) :
    pde_(PDE),
    rootPatch_(rootPatch),
    p4est_(p4est)
        {}

void FISHPACKHPSMethod::setupStage() {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Begin FISHPACK-HPS Setup Stage");

    // Initialize quadtree
    this->quadtree = new FISHPACKQuadtree(p4est_);
    quadtree->build(rootPatch_);

    // Set p4est ID (leaf level IDs)
    int currentID = 0;
    quadtree->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) patch.ID = currentID++;
        else patch.ID = -1;
    });

    // Set D2N on leaf
    quadtree->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) {
            FISHPACKFVSolver solver;
            if (std::get<bool>(app.options["cache-operators"])) {
                if (!matrixCache.contains("T_leaf")) {
                    // T for leaf patch is not set; build from patch solver
                    matrixCache["T_leaf"] = solver.buildD2N(patch.grid);
                }
                patch.T = matrixCache["T_leaf"];
            }
            else {
                patch.T = solver.buildD2N(patch.grid);
            }
        }
    });

    app.log("End FISHPACK-HPS Setup Stage");

}

void FISHPACKHPSMethod::merge4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Merging:");
    app.log("alpha = %i", alpha.ID);
    app.log("beta = %i", beta.ID);
    app.log("gamma = %i", gamma.ID);
    app.log("omega = %i", omega.ID);

    // Check for adaptivity
    std::vector<FISHPACKPatch*> patches = {&alpha, &beta, &gamma, &omega};
    Vector<int> tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);

    for (auto i = 0; i < 4; i++) {
        while (tags[i]-- > 0) {
            patches[i]->coarsen();
            patches[i] = patches[i]->coarser;
        }
    }

    // Perform the merge
    createIndexSets_(tau, alpha, beta, gamma, omega);
    // reorderT_(tau, alpha, beta, gamma, omega);
    createMatrixBlocks_(tau, alpha, beta, gamma, omega);
    mergeX_(tau, alpha, beta, gamma, omega);
    mergeS_(tau, alpha, beta, gamma, omega);
    mergeT_(tau, alpha, beta, gamma, omega);
    reorderOperators_(tau, alpha, beta, gamma, omega);
    mergePatch_(tau, alpha, beta, gamma, omega);

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

Vector<int> FISHPACKHPSMethod::tagPatchesForCoarsening_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    std::vector<FISHPACKPatch*> patches = {&alpha, &beta, &gamma, &omega};
    std::vector<int> gens(4);
    std::vector<int> tags(4);

    for (auto i = 0; i < 4; i++) gens[i] = static_cast<int>(log2(patches[i]->nPatchSideVector[EAST])); // 1 = EAST
    int minGens = *std::min_element(gens.begin(), gens.end());
    for (auto i = 0; i < 4; i++) tags[i] = gens[i] - minGens;

    return {tags};

}

void FISHPACKHPSMethod::createIndexSets_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {
    
    int nSide = alpha.grid.nPointsX();

    Vector<int> I_W = vectorRange(0, nSide-1);
    Vector<int> I_E = vectorRange(nSide, 2*nSide - 1);
    Vector<int> I_S = vectorRange(2*nSide, 3*nSide - 1);
    Vector<int> I_N = vectorRange(3*nSide, 4*nSide - 1);

    IS_alpha_beta_ = I_E;
    IS_alpha_gamma_ = I_N;
    IS_alpha_tau_ = concatenate({I_W, I_S});
    
    IS_beta_alpha_ = I_W;
    IS_beta_omega_ = I_N;
    IS_beta_tau_ = concatenate({I_E, I_S});
    
    IS_gamma_alpha_ = I_S;
    IS_gamma_omega_ = I_E;
    IS_gamma_tau_ = concatenate({I_W, I_N});

    IS_omega_beta_ = I_S;
    IS_omega_gamma_ = I_W;
    IS_omega_tau_ = concatenate({I_E, I_N});

    return;

}

void FISHPACKHPSMethod::reorderT_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    int nSide = alpha.grid.nPointsX();
    Vector<int> pi_alpha = {0, 2, 3, 1};
    Vector<int> pi_beta = {1, 2, 3, 0};
    Vector<int> pi_gamma = {0, 3, 2, 1};
    Vector<int> pi_omega = {1, 3, 2, 0};
    Vector<int> blockSizes(4, nSide);

    alpha.T = alpha.T.blockPermute(pi_alpha, pi_alpha, blockSizes, blockSizes);
    beta.T = beta.T.blockPermute(pi_beta, pi_beta, blockSizes, blockSizes);
    gamma.T = gamma.T.blockPermute(pi_gamma, pi_gamma, blockSizes, blockSizes);
    omega.T = omega.T.blockPermute(pi_omega, pi_omega, blockSizes, blockSizes);

}

void FISHPACKHPSMethod::createMatrixBlocks_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    Matrix<double>& T_alpha = alpha.T;
    Matrix<double>& T_beta = beta.T;
    Matrix<double>& T_gamma = gamma.T;
    Matrix<double>& T_omega = omega.T;

    // Blocks for X_tau
    T_ag_ag = T_alpha(IS_alpha_gamma_, IS_alpha_gamma_);
    T_ga_ga = T_gamma(IS_gamma_alpha_, IS_gamma_alpha_);
    T_ag_ab = T_alpha(IS_alpha_gamma_, IS_alpha_beta_);
    T_ga_go = T_gamma(IS_gamma_alpha_, IS_gamma_omega_);
    T_bo_bo = T_beta(IS_beta_omega_, IS_beta_omega_);
    T_ob_ob = T_omega(IS_omega_beta_, IS_omega_beta_);
    T_bo_ba = T_beta(IS_beta_omega_, IS_beta_alpha_);
    T_ob_og = T_omega(IS_omega_beta_, IS_omega_gamma_);
    T_ab_ag = T_alpha(IS_alpha_beta_, IS_alpha_gamma_);
    T_ba_bo = T_beta(IS_beta_alpha_, IS_beta_omega_);
    T_ab_ab = T_alpha(IS_alpha_beta_, IS_alpha_beta_);
    T_ba_ba = T_beta(IS_beta_alpha_, IS_beta_alpha_);
    T_go_ga = T_gamma(IS_gamma_omega_, IS_gamma_alpha_);
    T_og_ob = T_omega(IS_omega_gamma_, IS_omega_beta_);
    T_go_go = T_gamma(IS_gamma_omega_, IS_gamma_omega_);
    T_og_og = T_omega(IS_omega_gamma_, IS_omega_gamma_);

    // std::cout << "T_ag_ag = " << T_ag_ag << std::endl;

    // Blocks for S_tau
    T_ag_at = T_alpha(IS_alpha_gamma_, IS_alpha_tau_);
    T_ga_gt = T_gamma(IS_gamma_alpha_, IS_gamma_tau_);
    T_bo_bt = T_beta(IS_beta_omega_, IS_beta_tau_);
    T_ob_ot = T_omega(IS_omega_beta_, IS_omega_tau_);
    T_ab_at = T_alpha(IS_alpha_beta_, IS_alpha_tau_);
    T_ba_bt = T_beta(IS_beta_alpha_, IS_beta_tau_);
    T_go_gt = T_gamma(IS_gamma_omega_, IS_gamma_tau_);
    T_og_ot = T_omega(IS_omega_gamma_, IS_omega_tau_);

    // Blocks for T_tau
    T_at_at = T_alpha(IS_alpha_tau_, IS_alpha_tau_);
    T_bt_bt = T_beta(IS_beta_tau_, IS_beta_tau_);
    T_gt_gt = T_gamma(IS_gamma_tau_, IS_gamma_tau_);
    T_ot_ot = T_omega(IS_omega_tau_, IS_omega_tau_);
    T_at_ag = T_alpha(IS_alpha_tau_, IS_alpha_gamma_);
    T_at_ab = T_alpha(IS_alpha_tau_, IS_alpha_beta_);
    T_bt_bo = T_beta(IS_beta_tau_, IS_beta_omega_);
    T_bt_ba = T_beta(IS_beta_tau_, IS_beta_alpha_);
    T_gt_ga = T_gamma(IS_gamma_tau_, IS_gamma_alpha_);
    T_gt_go = T_gamma(IS_gamma_tau_, IS_gamma_omega_);
    T_ot_ob = T_omega(IS_omega_tau_, IS_omega_beta_);
    T_ot_og = T_omega(IS_omega_tau_, IS_omega_gamma_);

    // Negate blocks that need it
    T_ga_go = -T_ga_go;
    T_ob_og = -T_ob_og;
    T_ba_bo = -T_ba_bo;
    T_og_ob = -T_og_ob;
    T_ga_gt = -T_ga_gt;
    T_ob_ot = -T_ob_ot;
    T_ba_bt = -T_ba_bt;
    T_og_ot = -T_og_ot;

    // T_bt_bo = -T_bt_bo;
    // T_bt_bt = -T_bt_bt;

    return;

}

void FISHPACKHPSMethod::mergeX_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Create diagonals
    Matrix<double> T_diag1 = T_ag_ag - T_ga_ga;
    Matrix<double> T_diag2 = T_bo_bo - T_ob_ob;
    Matrix<double> T_diag3 = T_ab_ab - T_ba_ba;
    Matrix<double> T_diag4 = T_go_go - T_og_og;
    std::vector<Matrix<double>> diag = {T_diag1, T_diag2, T_diag3, T_diag4};

    // Create row and column block index starts
    std::vector<std::size_t> rowStarts = { 0, T_diag1.nRows(), T_diag1.nRows() + T_diag2.nRows(), T_diag1.nRows() + T_diag2.nRows() + T_diag3.nRows() };
    std::vector<std::size_t> colStarts = { 0, T_diag1.nCols(), T_diag1.nCols() + T_diag2.nCols(), T_diag1.nCols() + T_diag2.nCols() + T_diag3.nCols() };
    
    // Create matrix and set blocks
    tau.X = blockDiagonalMatrix(diag);
    tau.X.setBlock(rowStarts[0], colStarts[2], T_ag_ab);
    tau.X.setBlock(rowStarts[0], colStarts[3], T_ga_go);
    tau.X.setBlock(rowStarts[1], colStarts[2], T_bo_ba);
    tau.X.setBlock(rowStarts[1], colStarts[3], T_ob_og);
    tau.X.setBlock(rowStarts[2], colStarts[0], T_ab_ag);
    tau.X.setBlock(rowStarts[2], colStarts[1], T_ba_bo);
    tau.X.setBlock(rowStarts[3], colStarts[0], T_go_ga);
    tau.X.setBlock(rowStarts[3], colStarts[1], T_og_ob);

    std::cout << "alpha.T = " << alpha.T << std::endl;
    std::cout << "tau.X = " << tau.X << std::endl;

    return;

}

void FISHPACKHPSMethod::mergeS_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Create right hand side
    std::size_t nRows = T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() + T_go_gt.nRows();
    std::size_t nCols = T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() + T_ob_ot.nCols();
    Matrix<double> S_RHS(nRows, nCols, 0);

    std::vector<std::size_t> rowStarts = { 0, T_ag_at.nRows(), T_ag_at.nRows() + T_bo_bt.nRows(), T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() };
    std::vector<std::size_t> colStarts = { 0, T_ag_at.nCols(), T_ag_at.nCols() + T_bo_bt.nCols(), T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() };

    S_RHS.setBlock(rowStarts[0], colStarts[0], T_ag_at);
    S_RHS.setBlock(rowStarts[0], colStarts[2], T_ga_gt);
    S_RHS.setBlock(rowStarts[1], colStarts[1], T_bo_bt);
    S_RHS.setBlock(rowStarts[1], colStarts[3], T_ob_ot);
    S_RHS.setBlock(rowStarts[2], colStarts[0], T_ab_at);
    S_RHS.setBlock(rowStarts[2], colStarts[1], T_ba_bt);
    S_RHS.setBlock(rowStarts[3], colStarts[2], T_go_gt);
    S_RHS.setBlock(rowStarts[3], colStarts[3], T_og_ot);
    
    // Solve to set S_tau
    tau.S = solve(tau.X, S_RHS);

    std::cout << "S_RHS = " << S_RHS << std::endl;
    std::cout << "tau.S = " << tau.S << std::endl;

    return;

}

void FISHPACKHPSMethod::mergeT_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Create left hand side
    std::vector<Matrix<double>> diag = {T_at_at, T_bt_bt, T_gt_gt, T_ot_ot};
    Matrix<double> T_LHS = blockDiagonalMatrix(diag);

    // Create right hand side
    std::size_t nRows = T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() + T_ot_ob.nRows();
    std::size_t nCols = T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() + T_gt_go.nCols();
    Matrix<double> T_RHS(nRows, nCols, 0);

    std::vector<std::size_t> rowStarts = { 0, T_at_ag.nRows(), T_at_ag.nRows() + T_bt_bo.nRows(), T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() };
    std::vector<std::size_t> colStarts = { 0, T_at_ag.nCols(), T_at_ag.nCols() + T_bt_bo.nCols(), T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() };

    T_RHS.setBlock(rowStarts[0], colStarts[0], T_at_ag);
    T_RHS.setBlock(rowStarts[0], colStarts[2], T_at_ab);
    T_RHS.setBlock(rowStarts[1], colStarts[1], T_bt_bo);
    T_RHS.setBlock(rowStarts[1], colStarts[2], T_bt_ba);
    T_RHS.setBlock(rowStarts[2], colStarts[0], T_gt_ga);
    T_RHS.setBlock(rowStarts[2], colStarts[3], T_gt_go);
    T_RHS.setBlock(rowStarts[3], colStarts[1], T_ot_ob);
    T_RHS.setBlock(rowStarts[3], colStarts[3], T_ot_og);

    Matrix<double> T_RHS2 = T_RHS * tau.S;

    // Compute and set T_tau
    tau.T = T_LHS - T_RHS2;

    std::cout << "T_LHS = " << T_LHS << std::endl;
    std::cout << "T_RHS = " << T_RHS << std::endl;
    std::cout << "T_RHS2 = " << T_RHS2 << std::endl;
    std::cout << "tau.T = " << tau.T << std::endl;

    return;

}

void FISHPACKHPSMethod::reorderOperators_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Form permutation vector and block sizes
    int nSide = alpha.grid.nPointsX();
    Vector<int> pi_noChange = {0, 1, 2, 3};
    // Vector<int> pi_WESN = {0, 4, 2, 5, 1, 6, 3, 7};
    Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
    Vector<int> blockSizes1(4, nSide);
    Vector<int> blockSizes2(8, nSide);

    // Permute S and T
    tau.S = tau.S.blockPermute(pi_noChange, pi_WESN, blockSizes1, blockSizes2);
    tau.T = tau.T.blockPermute(pi_WESN, pi_WESN, blockSizes2, blockSizes2);

    std::cout << "tau.S PERMUTED = " << tau.S << std::endl;
    std::cout << "tau.T PERMUTED = " << tau.T << std::endl;

}

void FISHPACKHPSMethod::mergePatch_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    FISHPACKFVGrid mergedGrid(alpha.grid.nPointsX() + beta.grid.nPointsX(), alpha.grid.nPointsY() + gamma.grid.nPointsY(), alpha.grid.xLower(), beta.grid.xUpper(), alpha.grid.yLower(), gamma.grid.yUpper());
    tau.grid = mergedGrid;
    tau.level = alpha.level - 1;
    tau.isLeaf = false;
    tau.nPatchSideVector = {
        alpha.nPatchSideVector[WEST] + gamma.nPatchSideVector[WEST],
        beta.nPatchSideVector[EAST] + omega.nPatchSideVector[EAST],
        alpha.nPatchSideVector[SOUTH] + beta.nPatchSideVector[SOUTH],
        gamma.nPatchSideVector[NORTH] + omega.nPatchSideVector[NORTH]
    };
    tau.nCellsLeaf = alpha.nCellsLeaf;

}

} // NAMESPACE : FISHPACK

} // NAMESPACE : EllipticSolver