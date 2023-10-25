#include "FISHPACK.hpp"

namespace EllipticForest {

namespace FISHPACK {

#define DTN_OPTIMIZE 1

// ---=====================---
// FISHPACK Finite Volume Grid
// ---=====================---

FISHPACKFVGrid::FISHPACKFVGrid() :
    nx_(0),
    ny_(0),
    xLower_(-1.0),
    xUpper_(1.0),
    yLower_(-1.0),
    yUpper_(1.0),
    dx_(0),
    dy_(0)
        {}

FISHPACKFVGrid::FISHPACKFVGrid(std::size_t nx, std::size_t ny, double xLower, double xUpper, double yLower, double yUpper) :
        nx_(nx),
        ny_(ny),
        xLower_(xLower),
        xUpper_(xUpper),
        yLower_(yLower),
        yUpper_(yUpper),
        dx_((xUpper - xLower) / nx),
        dy_((yUpper - yLower) / ny),
        xPoints_(nx_),
        yPoints_(ny_) {

        for (auto i = 0; i < nx_; i++) {
            xPoints_[i] = operator()(XDIM, i);
        }
        for (auto j = 0; j < ny_; j++) {
            yPoints_[j] = operator()(YDIM, j);
        }

    }

std::string FISHPACKFVGrid::name() { return name_; }

std::size_t FISHPACKFVGrid::nx() { return nx_; }

std::size_t FISHPACKFVGrid::ny() { return ny_; }

double FISHPACKFVGrid::xLower() { return xLower_; }

double FISHPACKFVGrid::xUpper() { return xUpper_; }

double FISHPACKFVGrid::yLower() { return yLower_; }

double FISHPACKFVGrid::yUpper() { return yUpper_; }

double FISHPACKFVGrid::dx() { return dx_; }

double FISHPACKFVGrid::dy() { return dy_; }

double FISHPACKFVGrid::operator()(std::size_t DIM, std::size_t index)  {
    if (DIM == XDIM) {
        if (index >= nx_ || index < 0) {
            std::string errorMessage = "[EllipticForest::FISHPACKFVGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tnx = " + std::to_string(nx_) + "\n";
            std::cerr << errorMessage << std::endl;
            throw std::out_of_range(errorMessage);
        }
        return (xLower_ + dx_/2) + index*dx_;
    }
    else if (DIM == YDIM) {
        if (index >= ny_ || index < 0) {
            std::string errorMessage = "[EllipticForest::FISHPACKFVGrid::operator()] `index` is out of range:\n";
            errorMessage += "\tindex = " + std::to_string(index) + "\n";
            errorMessage += "\tny = " + std::to_string(ny_) + "\n";
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

Vector<double>& FISHPACKFVGrid::xPoints() { return xPoints_; }
Vector<double>& FISHPACKFVGrid::yPoints() { return yPoints_; }

std::string FISHPACKFVGrid::getWholeExtent() {
    return "0 " + std::to_string(nx_-1) + " 0 " + std::to_string(ny_-1) + " 0 0";
}

std::string FISHPACKFVGrid::getExtent() {
    return "0 " + std::to_string(nx_-1) + " 0 " + std::to_string(ny_-1) + " 0 0";
}

// ---=============================---
// FISHPACK Finite Volume Patch Solver
// ---=============================---

FISHPACKFVSolver::FISHPACKFVSolver() {}

FISHPACKFVSolver::FISHPACKFVSolver(double lambda) {
    this->lambda = lambda;
}

std::string FISHPACKFVSolver::name() {
    return "FISHPACK90Solver";
}

Vector<double> FISHPACKFVSolver::solve(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // Unpack Dirichlet Data
	int nSide = grid.nx();
	Vector<double> gWest = dirichletData.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichletData.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichletData.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichletData.getSegment(3*nSide, nSide);

	// Transpose RHS data for FORTRAN call
	Vector<double> fT(nSide * nSide);
	for (int i = 0; i < grid.nx(); i++) {
		for (int j = 0; j < grid.ny(); j++) {
			fT[i + j*nSide] = rhsData[j + i*nSide];
		}
	}

	// Setup FORTRAN call to FISHPACK
	double A = grid.xLower();
	double B = grid.xUpper();
	int M = grid.nx();
	int MBDCND = 1;
	double* BDA = gWest.dataPointer();
	double* BDB = gEast.dataPointer();
	double C = grid.yLower();
	double D = grid.yUpper();
	int N = grid.ny();
	int NBDCND = 1;
	double* BDC = gSouth.dataPointer();
	double* BDD = gNorth.dataPointer();
	double ELMBDA = this->lambda; // @TODO: Implement or get lambda value
	double* F = fT.dataPointer();
    // double* F = rhsData.dataPointer();
	int IDIMF = M;
	double PERTRB;
	int IERROR;
	// int WSIZE = 13*M + 4*N + M*((int)log2(N));
	// double* W = (double*) malloc(WSIZE*sizeof(double));

	// Make FORTRAN call to FISHPACK
	// hstcrtt_(&A, &B, &M, &MBDCND, BDA, BDB,
	// 		&C, &D, &N, &NBDCND, BDC, BDD,
	// 		&ELMBDA, F, &IDIMF, &PERTRB, &IERROR, W);
    hstcrt_(&A, &B, &M, &MBDCND, BDA, BDB,
			&C, &D, &N, &NBDCND, BDC, BDD,
			&ELMBDA, F, &IDIMF, &PERTRB, &IERROR);
	if (IERROR != 0 && IERROR != 6) {
		std::cerr << "[EllipticForest::FISHPACK::FISHPACKFVSolver::solve] WARNING: call to hstcrt_ returned non-zero error value: IERROR = " << IERROR << std::endl;
	}
    if (fabs(PERTRB) > 1e-8) {
        std::cerr << "[EllipticForest::FISHPACK::FISHPACKFVSolver::solve] WARNING: PERTRB value from FISHPACK is non-zero: PERTRB = " << PERTRB << std::endl;
    }

	// Move FISHPACK solution into Vector for output
	Vector<double> solution(grid.nx() * grid.ny());
	for (int i = 0; i < grid.nx(); i++) {
		for (int j = 0; j < grid.ny(); j++) {
			// solution[j + i*nSide] = F[i + j*nSide];
            solution[j + i*nSide] = F[i + j*nSide];
		}
	}

    // free(W);

	return solution; // return rhsData;

}

Vector<double> FISHPACKFVSolver::mapD2N(PatchGridBase<double>& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // Unpack grid data
	int nSide = grid.nx();

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
        int idxW = j;
        int idxE = (nSide-1)*nSide + j;
		uWest[j] = u[idxW];
		uEast[j] = u[idxE];
	}
	for (int i = 0; i < nSide; i++) {
        int idxS = i*nSide;
        int idxN = (i+1)*nSide - 1;
		uSouth[i] = u[idxS];
		uNorth[i] = u[idxN];
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

    std::size_t N = grid.nx();
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
		col_j = this->mapD2N(grid, e_hat_j, f_zero);
		T.setColumn(j, col_j);
		e_hat_j[j] = 0.0;
	}
#endif
	return T;

}

// Vector<double> FISHPACKFVSolver::rhsData(PatchGridBase<double>& grid) {
//     Vector<double> f(grid.nx() * grid.ny());
//     for (auto i = 0; i < grid.nx(); i++) {
//         double x = grid(XDIM, i);
//         for (auto j = 0; j < grid.ny(); j++) {
//             double y = grid(YDIM, j);
//             int index = j + i*grid.ny();
//             f[index] = pde.f(x, y);
//         }
//     }
//     return f;
// }

// ---======================---
// FISHPACK Finite Volume Patch
// ---======================---

FISHPACKPatch::FISHPACKPatch() {}

FISHPACKPatch::FISHPACKPatch(FISHPACKFVGrid grid) :
    grid_(grid)
        {}

FISHPACKPatch::~FISHPACKPatch() {}

std::string FISHPACKPatch::name() { return "FISHPACKPatch"; }

FISHPACKFVGrid& FISHPACKPatch::grid() { return grid_; }

// FISHPACKPatch FISHPACKPatch::buildChild(std::size_t childIndex) {

//     std::size_t nx = grid_.nx();
//     std::size_t ny = grid_.ny();
//     double xMid = (grid_.xLower() + grid_.xUpper()) / 2.0;
//     double yMid = (grid_.yLower() + grid_.yUpper()) / 2.0;
//     double xLower, xUpper, yLower, yUpper;
//     switch (childIndex) {
//         case 0:
//             xLower = grid_.xLower();
//             xUpper = xMid;
//             yLower = grid_.yLower();
//             yUpper = yMid;
//             break;
//         case 1:
//             xLower = xMid;
//             xUpper = grid_.xUpper();
//             yLower = grid_.yLower();
//             yUpper = yMid;
//             break;
//         case 2:
//             xLower = grid_.xLower();
//             xUpper = xMid;
//             yLower = yMid;
//             yUpper = grid_.yUpper();
//             break;
//         case 3:
//             xLower = xMid;
//             xUpper = grid_.xUpper();
//             yLower = yMid;
//             yUpper = grid_.yUpper();
//             break;
//     }
//     FISHPACKFVGrid childGrid(nx, ny, xLower, xUpper, yLower, yUpper);

//     FISHPACKPatch childPatch(childGrid);
//     childPatch.level = level + 1;
//     childPatch.isLeaf = true;
//     isLeaf = false;

//     return childPatch;
// }

Matrix<double>& FISHPACKPatch::matrixX() { return X; }

Matrix<double>& FISHPACKPatch::matrixH() { return H; }

Matrix<double>& FISHPACKPatch::matrixS() { return S; }

Matrix<double>& FISHPACKPatch::matrixT() { return T; }

Vector<double>& FISHPACKPatch::vectorU() { return u; }

Vector<double>& FISHPACKPatch::vectorG() { return g; }

Vector<double>& FISHPACKPatch::vectorV() { return v; }

Vector<double>& FISHPACKPatch::vectorF() { return f; }

Vector<double>& FISHPACKPatch::vectorH() { return h; }

Vector<double>& FISHPACKPatch::vectorW() { return w; }

std::string FISHPACKPatch::str() {

    std::string res;

    // res += "globalID = " + std::to_string(globalID) + "\n";
    // res += "leafID = " + std::to_string(leafID) + "\n";
    // res += "level = " + std::to_string(level) + "\n";
    // res += "isLeaf = " + std::to_string(isLeaf) + "\n";
    res += "nCoarsens = " + std::to_string(nCoarsens) + "\n";

    res += "grid:\n";
    res += "  nx = " + std::to_string(grid().nx()) + "\n";
    res += "  ny = " + std::to_string(grid().ny()) + "\n";
    res += "  xLower = " + std::to_string(grid().xLower()) + "\n";
    res += "  xUpper = " + std::to_string(grid().xUpper()) + "\n";
    res += "  yLower = " + std::to_string(grid().yLower()) + "\n";
    res += "  yUpper = " + std::to_string(grid().yUpper()) + "\n";

    res += "data:\n";
    res += "  X = [" + std::to_string(matrixX().nRows()) + ", " + std::to_string(matrixX().nCols()) + "]\n";
    res += "  S = [" + std::to_string(matrixS().nRows()) + ", " + std::to_string(matrixS().nCols()) + "]\n";
    res += "  T = [" + std::to_string(matrixT().nRows()) + ", " + std::to_string(matrixT().nCols()) + "]\n";
    res += "  u = [" + std::to_string(vectorU().size()) + "]\n";
    res += "  g = [" + std::to_string(vectorG().size()) + "]\n";
    res += "  v = [" + std::to_string(vectorV().size()) + "]\n";
    res += "  f = [" + std::to_string(vectorF().size()) + "]\n";
    res += "  h = [" + std::to_string(vectorH().size()) + "]\n";
    res += "  w = [" + std::to_string(vectorW().size()) + "]\n";

    return res;

}

double FISHPACKPatch::dataSize() {

    double BYTE_2_MEGABYTE = 1024*1024;
    double size_MB = (4*sizeof(int) + sizeof(bool)) / BYTE_2_MEGABYTE;

    size_MB += (T.nRows() * T.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (H.nRows() * H.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (S.nRows() * S.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (X.nRows() * X.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;

    size_MB += (u.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (g.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (v.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (f.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (h.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (w.size() * sizeof(double)) / BYTE_2_MEGABYTE;

    return size_MB;

}

FISHPACKPatchNodeFactory::FISHPACKPatchNodeFactory() {}

FISHPACKPatchNodeFactory::FISHPACKPatchNodeFactory(MPI_Comm comm) :
    MPIObject(comm)
        {}

Node<FISHPACKPatch>* FISHPACKPatchNodeFactory::createNode(FISHPACKPatch data, std::string path, int level, int pfirst, int plast) {
    return new Node<FISHPACKPatch>(this->getComm(), data, path, level, pfirst, plast);
}

Node<FISHPACKPatch>* FISHPACKPatchNodeFactory::createChildNode(Node<FISHPACKPatch>* parentNode, int siblingID, int pfirst, int plast) {
    
    // Get parent grid info
    auto& parentGrid = parentNode->data.grid();
    int nx = parentGrid.nx();
    int ny = parentGrid.ny();
    double xLower = parentGrid.xLower();
    double xUpper = parentGrid.xUpper();
    double xMid = (xLower + xUpper) / 2.0;
    double yLower = parentGrid.yLower();
    double yUpper = parentGrid.yUpper();
    double yMid = (yLower + yUpper) / 2.0;

    // Create child grid
    FISHPACKFVGrid childGrid;
    switch (siblingID)
    {
    case 0:
        // Lower left
        childGrid = FISHPACKFVGrid(nx, ny, xLower, xMid, yLower, yMid);
        break;
    case 1:
        // Lower right
        childGrid = FISHPACKFVGrid(nx, ny, xMid, xUpper, yLower, yMid);
        break;
    case 2:
        // Upper left
        childGrid = FISHPACKFVGrid(nx, ny, xLower, xMid, yMid, yUpper);
        break;
    case 3:
        // Upper right
        childGrid = FISHPACKFVGrid(nx, ny, xMid, xUpper, yMid, yUpper);
        break;
    default:
        break;
    }

    // Create child patch
    FISHPACKPatch childPatch(childGrid);

    // Create child node
    std::string path = parentNode->path + std::to_string(siblingID);
    int level = parentNode->level + 1;
    return new Node<FISHPACKPatch>(this->getComm(), childPatch, path, level, pfirst, plast);
    
}

Node<FISHPACKPatch>* FISHPACKPatchNodeFactory::createParentNode(std::vector<Node<FISHPACKPatch>*> childNodes, int pfirst, int plast) {

    // Create parent grid
    int nx = childNodes[0]->data.grid().nx();
    int ny = childNodes[0]->data.grid().ny();
    double xLower = childNodes[0]->data.grid().xLower();
    double xUpper = childNodes[1]->data.grid().xUpper();
    double yLower = childNodes[0]->data.grid().yLower();
    double yUpper = childNodes[2]->data.grid().yUpper();
    FISHPACKFVGrid parentGrid(nx, ny, xLower, xUpper, yLower, yUpper);

    // Create parent patch
    FISHPACKPatch parentPatch(parentGrid);

    // Create parent node
    std::string path = childNodes[0]->path.substr(0, childNodes[0]->path.length()-1);
    int level = childNodes[0]->level - 1;
    return new Node<FISHPACKPatch>(this->getComm(), parentPatch, path, level, pfirst, plast);

}

} // NAMESPACE : FISHPACK

namespace MPI {

template<>
int broadcast(FISHPACK::FISHPACKFVGrid& grid, int root, MPI_Comm comm) {
    int nx = grid.nx();
    int ny = grid.ny();
    double xLower = grid.xLower();
    double xUpper = grid.xUpper();
    double yLower = grid.yLower();
    double yUpper = grid.yUpper();
    broadcast(nx, root, comm);
    broadcast(ny, root, comm);
    broadcast(xLower, root, comm);
    broadcast(xUpper, root, comm);
    broadcast(yLower, root, comm);
    broadcast(yUpper, root, comm);
    int rank; MPI_Comm_rank(comm, &rank);
    if (rank != root) grid = FISHPACK::FISHPACKFVGrid(nx, ny, xLower, xUpper, yLower, yUpper);
    return 1;
}

template<>
int broadcast(FISHPACK::FISHPACKPatch& patch, int root, MPI_Comm comm) {
    broadcast(patch.nCoarsens, root, comm);
    broadcast(patch.grid(), root, comm);
    broadcast(patch.matrixX(), root, comm);
    broadcast(patch.matrixH(), root, comm);
    broadcast(patch.matrixS(), root, comm);
    broadcast(patch.matrixT(), root, comm);
    broadcast(patch.vectorU(), root, comm);
    broadcast(patch.vectorG(), root, comm);
    broadcast(patch.vectorV(), root, comm);
    broadcast(patch.vectorF(), root, comm);
    broadcast(patch.vectorH(), root, comm);
    broadcast(patch.vectorW(), root, comm);
    return 1;
}

} // NAMESPACE : MPI

} // NAMESPACE : EllipticSolver