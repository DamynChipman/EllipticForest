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
        dy_((yUpper - yLower) / nPointsY),
        xPoints_(nPointsX_),
        yPoints_(nPointsY_) {

        for (auto i = 0; i < nPointsX_; i++) {
            xPoints_[i] = operator()(XDIM, i);
        }
        for (auto j = 0; j < nPointsY_; j++) {
            yPoints_[j] = operator()(YDIM, j);
        }

    }

std::string FISHPACKFVGrid::name() { return name_; }

std::size_t FISHPACKFVGrid::nPointsX() { return nPointsX_; }

std::size_t FISHPACKFVGrid::nPointsY() { return nPointsY_; }

double FISHPACKFVGrid::xLower() { return xLower_; }

double FISHPACKFVGrid::xUpper() { return xUpper_; }

double FISHPACKFVGrid::yLower() { return yLower_; }

double FISHPACKFVGrid::yUpper() { return yUpper_; }

double FISHPACKFVGrid::dx() { return dx_; }

double FISHPACKFVGrid::dy() { return dy_; }

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

Vector<double>& FISHPACKFVGrid::xPoints() { return xPoints_; }
Vector<double>& FISHPACKFVGrid::yPoints() { return yPoints_; }

std::string FISHPACKFVGrid::getWholeExtent() {
    return "0 " + std::to_string(nPointsX_-1) + " 0 " + std::to_string(nPointsY_-1) + " 0 0";
}

std::string FISHPACKFVGrid::getExtent() {
    return "0 " + std::to_string(nPointsX_-1) + " 0 " + std::to_string(nPointsY_-1) + " 0 0";
}

// ---=============================---
// FISHPACK Finite Volume Patch Solver
// ---=============================---

FISHPACKFVSolver::FISHPACKFVSolver() {
    EllipticForestApp& app = EllipticForestApp::getInstance();
    if (app.options.optionExists("lambda")) {
        lambda = std::get<double>(app.options["lambda"]);
    }
}

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
	double ELMBDA = lambda; // @TODO: Implement or get lambda value
	double* F = fT.dataPointer();
    // double* F = rhsData.dataPointer();
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
			// solution[j + i*nSide] = F[i + j*nSide];
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
		col_j = this->mapD2N(grid, e_hat_j, f_zero);
		T.setColumn(j, col_j);
		e_hat_j[j] = 0.0;
	}
#endif
	return T;

}

// Vector<double> FISHPACKFVSolver::rhsData(PatchGridBase<double>& grid) {
//     Vector<double> f(grid.nPointsX() * grid.nPointsY());
//     for (auto i = 0; i < grid.nPointsX(); i++) {
//         double x = grid(XDIM, i);
//         for (auto j = 0; j < grid.nPointsY(); j++) {
//             double y = grid(YDIM, j);
//             int index = j + i*grid.nPointsY();
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

FISHPACKPatch FISHPACKPatch::buildChild(std::size_t childIndex) {

    std::size_t nx = grid_.nPointsX();
    std::size_t ny = grid_.nPointsY();
    double xMid = (grid_.xLower() + grid_.xUpper()) / 2.0;
    double yMid = (grid_.yLower() + grid_.yUpper()) / 2.0;
    double xLower, xUpper, yLower, yUpper;
    switch (childIndex) {
        case 0:
            xLower = grid_.xLower();
            xUpper = xMid;
            yLower = grid_.yLower();
            yUpper = yMid;
            break;
        case 1:
            xLower = xMid;
            xUpper = grid_.xUpper();
            yLower = grid_.yLower();
            yUpper = yMid;
            break;
        case 2:
            xLower = grid_.xLower();
            xUpper = xMid;
            yLower = yMid;
            yUpper = grid_.yUpper();
            break;
        case 3:
            xLower = xMid;
            xUpper = grid_.xUpper();
            yLower = yMid;
            yUpper = grid_.yUpper();
            break;
    }
    FISHPACKFVGrid childGrid(nx, ny, xLower, xUpper, yLower, yUpper);

    FISHPACKPatch childPatch(childGrid);
    childPatch.level = level - 1;
    childPatch.isLeaf = true;
    isLeaf = false;

    return childPatch;
}

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

    res += "globalID = " + std::to_string(globalID) + "\n";
    res += "leafID = " + std::to_string(leafID) + "\n";
    res += "level = " + std::to_string(level) + "\n";
    res += "isLeaf = " + std::to_string(isLeaf) + "\n";
    res += "nCoarsens = " + std::to_string(nCoarsens) + "\n";

    res += "grid:\n";
    res += "  nx = " + std::to_string(grid().nPointsX()) + "\n";
    res += "  ny = " + std::to_string(grid().nPointsY()) + "\n";
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

// ---=========================---
// FISHPACK Finite Volume Quadtree
// ---=========================---
#if 0
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
    patch.globalID = globalIndices_[level][index];
    patch.grid = grid;
    patch.level = level;
    if (this->childIndices_[level][index] == -1) {
        patch.isLeaf = true;
    }
    else {
        patch.isLeaf = false;
    }

    return patch;

}

void FISHPACKQuadtree::toVTK(std::string filename) {

    XMLNode nodeVTKFile("VTKFile");
    nodeVTKFile.addAttribute("type", "PRectilinearGrid");
    nodeVTKFile.addAttribute("version", "0.1");
    nodeVTKFile.addAttribute("byte_order", "LittleEndian");

    XMLNode nodePRectilinearGrid("PRectilinearGrid");
    int nLeafPatches = 0;
    this->traversePreOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) nLeafPatches++;
    });
    int extent = nLeafPatches * this->data_[0].grid.nPointsX(); // <-- Maybe...?
    nodePRectilinearGrid.addAttribute("WholeExtent", "0 " + std::to_string(extent) + " 0 " + std::to_string(extent) + " 0 0");
    nodePRectilinearGrid.addAttribute("GhostLevel", "0");

    XMLNode nodePPointData("PPointData");
    nodePPointData.addAttribute("Scalars", "u f");

    XMLNode nodePPointDataArray1("PDataArray");
    nodePPointDataArray1.addAttribute("type", "Float32");
    nodePPointDataArray1.addAttribute("Name", "u");
    nodePPointDataArray1.addAttribute("NumberOfComponents", "1");

    XMLNode nodePPointDataArray2("PDataArray");
    nodePPointDataArray2.addAttribute("type", "Float32");
    nodePPointDataArray2.addAttribute("Name", "u");
    nodePPointDataArray2.addAttribute("NumberOfComponents", "1");

    XMLNode nodePCoordinates("PCoordinates");
    XMLNode nodeXCoordDataArray("PDataArray");
    nodeXCoordDataArray.addAttribute("type", "Float32");
    nodeXCoordDataArray.addAttribute("NumberOfComponents", "1");
    XMLNode nodeYCoordDataArray("PDataArray");
    nodeYCoordDataArray.addAttribute("type", "Float32");
    nodeYCoordDataArray.addAttribute("NumberOfComponents", "1");
    XMLNode nodeZCoordDataArray("PDataArray");
    nodeZCoordDataArray.addAttribute("type", "Float32");
    nodeZCoordDataArray.addAttribute("NumberOfComponents", "1");

    std::vector<XMLNode> nodesPiece;
    this->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) {
            RectilinearGridVTK gridVTK;
            gridVTK.buildMesh(patch.grid, &patch.grid.xPoints(), &patch.grid.yPoints(), nullptr);
            patch.u.name() = "u";
            patch.f.name() = "f";
            gridVTK.addPointData(patch.u);
            gridVTK.addPointData(patch.f);
            std::string pieceFilename = filename + "_" + std::to_string(patch.leafID) + ".vtr";
            gridVTK.toVTK(pieceFilename);

            XMLNode nodePiece("Piece");
            nodePiece.addAttribute("Extent", "0 " + std::to_string(patch.grid.nPointsX()-1) + " 0 " + std::to_string(patch.grid.nPointsY()-1) + " 0 0");
            nodePiece.addAttribute("Source", pieceFilename);
            nodesPiece.push_back(nodePiece);
        }
    });

    nodeVTKFile.addChild(nodePRectilinearGrid);
    nodePRectilinearGrid.addChild(nodePPointData);
        nodePPointData.addChild(nodePPointDataArray1);
        nodePPointData.addChild(nodePPointDataArray2);
    nodePRectilinearGrid.addChild(nodePCoordinates);
        nodePCoordinates.addChild(nodeXCoordDataArray);
        nodePCoordinates.addChild(nodeYCoordDataArray);
        nodePCoordinates.addChild(nodeZCoordDataArray);
    for (auto& node : nodesPiece) nodePRectilinearGrid.addChild(node);

    XMLTree xmlTree(nodeVTKFile);
    xmlTree.write(filename + ".pvtr");

}

// ---===========================---
// FISHPACK Finite Volume HPS Method
// ---===========================---

FISHPACKHPSMethod::FISHPACKHPSMethod(FISHPACKProblem& PDE, FISHPACKPatch& rootPatch, p4est_t* p4est) :
    pde_(PDE),
    rootPatch_(rootPatch),
    p4est_(p4est)
        {}

void FISHPACKHPSMethod::toVTK(std::string filename) {
    quadtree->toVTK(filename);
}

void FISHPACKHPSMethod::setupStage() {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Begin FISHPACK-HPS Setup Stage");

    // Initialize quadtree
    this->quadtree = new FISHPACKQuadtree(p4est_);
    quadtree->build(rootPatch_);

    // Set p4est ID (leaf level IDs)
    int currentID = 0;
    quadtree->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) {
            patch.leafID = currentID++;
        }
        else patch.leafID = -1;
    });

    // Set D2N on leaf
    quadtree->traversePostOrder([&](FISHPACKPatch& patch){
        if (patch.isLeaf) {
            FISHPACKFVSolver solver;
            if (std::get<bool>(app.options["cache-operators"])) {
                if (!matrixCache.contains("T_leaf")) {
                    // T for leaf patch is not set; build from patch solver
                    matrixCache["T_leaf"] = solver.buildD2N(patch.grid); // BUG/TODO: Need to scale T by leaf level
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
    // app.log("Merging:");
    // app.log("  alpha = %i", alpha.globalID);
    // app.log("  beta = %i", beta.globalID);
    // app.log("  gamma = %i", gamma.globalID);
    // app.log("  omega = %i", omega.globalID);
    // app.log("  tau = %i", tau.globalID);

    // Steps for the merge (private member functions)
    coarsen_(tau, alpha, beta, gamma, omega);
    createIndexSets_(tau, alpha, beta, gamma, omega);
    createMatrixBlocks_(tau, alpha, beta, gamma, omega);
    mergeX_(tau, alpha, beta, gamma, omega);
    mergeS_(tau, alpha, beta, gamma, omega);
    mergeT_(tau, alpha, beta, gamma, omega);
    reorderOperators_(tau, alpha, beta, gamma, omega);
    mergePatch_(tau, alpha, beta, gamma, omega);

}

void FISHPACKHPSMethod::upwards4to1(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    if (!std::get<bool>(app.options["homogeneous-rhs"])) {
        // app.log("Upwards:");
        // app.log("  alpha = %i", alpha.globalID);
        // app.log("  beta = %i", beta.globalID);
        // app.log("  gamma = %i", gamma.globalID);
        // app.log("  omega = %i", omega.globalID);
        // app.log("  tau = %i", tau.globalID);

        // Steps for the upwards stage (private member functions)
        coarsenUpwards_(tau, alpha, beta, gamma, omega);
        createIndexSets_(tau, alpha, beta, gamma, omega);
        createMatrixBlocks_(tau, alpha, beta, gamma, omega);
        mergeX_(tau, alpha, beta, gamma, omega);
        mergeW_(tau, alpha, beta, gamma, omega);
        mergeH_(tau, alpha, beta, gamma, omega);
        reorderOperatorsUpwards_(tau, alpha, beta, gamma, omega);
    }

}

void FISHPACKHPSMethod::split1to4(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    // app.log("Splitting:");
    // app.log("  tau = %i", tau.globalID);
    // app.log("  alpha = %i", alpha.globalID);
    // app.log("  beta = %i", beta.globalID);
    // app.log("  gamma = %i", gamma.globalID);
    // app.log("  omega = %i", omega.globalID);

    // Steps for the split (private member functions)
    uncoarsen_(tau, alpha, beta, gamma, omega);
    applyS_(tau, alpha, beta, gamma, omega);

}

void FISHPACKHPSMethod::setParticularData(FISHPACKPatch& patch) {

    if (patch.isLeaf) {

        // Create solver
        FISHPACKFVSolver solver;

        // Set RHS data on leaf patches
        FISHPACKFVGrid& grid = patch.grid;
        patch.f = Vector<double>(grid.nPointsX() * grid.nPointsY());
        for (auto i = 0; i < grid.nPointsX(); i++) {
            double x = grid(XDIM, i);
            for (auto j = 0; j < grid.nPointsY(); j++) {
                double y = grid(YDIM, j);
                int index = j + i*grid.nPointsY();
                patch.f[index] = pde_.f(x, y);
            }
        }

        // Set Neumann data for particular solution
        Vector<double> gZero(2*grid.nPointsX() + 2*grid.nPointsY(), 0);
        patch.h = solver.mapD2N(grid, gZero, patch.f);

    }

}

void FISHPACKHPSMethod::preSolveHook() {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    app.log("Setting Dirichlet data on the root patch");

    // Set Dirichlet data on root patch
    FISHPACKPatch& rootPatch = this->quadtree->root();
    FISHPACKFVGrid& grid = rootPatch.grid;
    std::size_t nBoundary = 2*grid.nPointsX() + 2*grid.nPointsY();
    rootPatch.g = Vector<double>(nBoundary);
    Vector<int> IS_West = vectorRange(0, grid.nPointsY() - 1);
    Vector<int> IS_East = vectorRange(grid.nPointsY(), 2*grid.nPointsY() - 1);
    Vector<int> IS_South = vectorRange(2*grid.nPointsY(), 2*grid.nPointsY() + grid.nPointsX() - 1);
    Vector<int> IS_North = vectorRange(2*grid.nPointsY() + grid.nPointsX(), 2*grid.nPointsY() + 2*grid.nPointsX() - 1);
    Vector<int> IS_WESN = concatenate({IS_West, IS_East, IS_South, IS_North});
    for (auto i = 0; i < nBoundary; i++) {
        std::size_t iSide = i % grid.nPointsX();
        double x, y;
        if (std::find(IS_West.data().begin(), IS_West.data().end(), i) != IS_West.data().end()) {
            x = grid.xLower();
            y = grid(YDIM, iSide);
            rootPatch.g[i] = pde_.u(x, y);
        }
        if (std::find(IS_East.data().begin(), IS_East.data().end(), i) != IS_East.data().end()) {
            x = grid.xUpper();
            y = grid(YDIM, iSide);
            rootPatch.g[i] = pde_.u(x, y);
        }
        if (std::find(IS_South.data().begin(), IS_South.data().end(), i) != IS_South.data().end()) {
            x = grid(XDIM, iSide);
            y = grid.yLower();
            rootPatch.g[i] = pde_.u(x, y);
        }
        if (std::find(IS_North.data().begin(), IS_North.data().end(), i) != IS_North.data().end()) {
            x = grid(XDIM, iSide);
            y = grid.yUpper();
            rootPatch.g[i] = pde_.u(x, y);
        }
    }

    return;

}

void FISHPACKHPSMethod::leafSolve(FISHPACKPatch& patch) {

    EllipticForestApp& app = EllipticForestApp::getInstance();
    if (patch.isLeaf) {
        FISHPACKFVSolver solver;
        if (std::get<bool>(app.options["homogeneous-rhs"])) {
            // Need to set RHS to zeros for patch solver b/c it won't be set already
            patch.f = Vector<double>(patch.grid.nPointsX() * patch.grid.nPointsY(), 0);
        }
        patch.u = Vector<double>(patch.grid.nPointsX()*patch.grid.nPointsY());
        Vector<double> u = solver.solve(patch.grid, patch.g, patch.f);
        for (auto i = 0; i < patch.grid.nPointsX(); i++) {
            for (auto j = 0; j < patch.grid.nPointsY(); j++) {
                int index = i + j*patch.grid.nPointsY();
                int index_T = j + i*patch.grid.nPointsY();
                double x = patch.grid(XDIM, i);
                double y = patch.grid(YDIM, j);
                patch.u[index] = u[index_T];
                patch.f[index] = pde_.f(x,y);
            }
        }
    }
        
}

Vector<int> FISHPACKHPSMethod::tagPatchesForCoarsening_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    std::vector<FISHPACKFVGrid*> grids = {&alpha.grid, &beta.grid, &gamma.grid, &omega.grid};
    std::vector<int> sides(4);
    std::vector<int> tags(4);

    for (auto i = 0; i < 4; i++) { sides[i] = grids[i]->nPointsX(); }   // Get vector of side lengths
    int minSide = *std::min_element(sides.begin(), sides.end());        // Get minimum side length
    for (auto i = 0; i < 4; i++) { tags[i] = (sides[i] / minSide) - 1; }      // Get tags based on side lengths

    return {tags};

}

void FISHPACKHPSMethod::coarsen_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Check for adaptivity
    std::vector<FISHPACKPatch*> patchPointers = {&alpha, &beta, &gamma, &omega};
    Vector<int> tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);
    int maxTag = *std::max_element(tags.data().begin(), tags.data().end());
    while (maxTag > 0) {
        for (auto i = 0; i < 4; i++) {
            if (tags[i] > 0) {
                FISHPACKFVGrid fineGrid = patchPointers[i]->grid;
                FISHPACKFVGrid coarseGrid = FISHPACKFVGrid(fineGrid.nPointsX()/2, fineGrid.nPointsY()/2, fineGrid.xLower(), fineGrid.xUpper(), fineGrid.yLower(), fineGrid.yUpper());
                // int nFine = fineGrid.nPointsX();
                // int nCoarse = coarseGrid.nPointsX();
        
                // InterpolationMatrixFine2Coarse<double> L21Side(nCoarse);
                // std::vector<Matrix<double>> L21Diagonals = {L21Side, L21Side, L21Side, L21Side};
                // Matrix<double> L21Patch = blockDiagonalMatrix(L21Diagonals);

                // InterpolationMatrixCoarse2Fine<double> L12Side(nFine);
                // std::vector<Matrix<double>> L12Diagonals = {L12Side, L12Side, L12Side, L12Side};
                // Matrix<double> L12Patch = blockDiagonalMatrix(L12Diagonals);

                // patchPointers[i]->T = L21Patch * patchPointers[i]->T;
                // patchPointers[i]->T = patchPointers[i]->T * L12Patch;
                FISHPACKFVSolver solver;
                patchPointers[i]->T = solver.buildD2N(coarseGrid);

                patchPointers[i]->grid = coarseGrid;
                patchPointers[i]->nCoarsens++;
            }
        }
        tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);
        maxTag = *std::max_element(tags.data().begin(), tags.data().end());
    }

    return;

}

void FISHPACKHPSMethod::createIndexSets_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {
    
    // Check that all children patches are the same size (should be handled from the coarsening step if not)
    int nAlpha = alpha.grid.nPointsX();
    int nBeta = beta.grid.nPointsX();
    int nGamma = gamma.grid.nPointsX();
    int nOmega = omega.grid.nPointsX();
    Vector<int> n = {nAlpha, nBeta, nGamma, nOmega};
    if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
        throw std::invalid_argument("[EllipticForest::FISHPACK::FISHPACKHPSMethod::createIndexSets_] Size of children patches are not the same; something probably went wrong with the coarsening...");
    }

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
    T_ag_at = -T_ag_at;
    T_bo_bt = -T_bo_bt;
    T_ab_at = -T_ab_at;
    T_go_gt = -T_go_gt;

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

    return;

}

void FISHPACKHPSMethod::mergeT_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Create left hand side
    std::vector<Matrix<double>> diag = {T_at_at, T_bt_bt, T_gt_gt, T_ot_ot};
    Matrix<double> T_LHS = blockDiagonalMatrix(diag);

    // Create right hand side
    std::size_t nRows = T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() + T_ot_ob.nRows();
    std::size_t nCols = T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() + T_gt_go.nCols();
    tau.H = Matrix<double>(nRows, nCols, 0);

    std::vector<std::size_t> rowStarts = { 0, T_at_ag.nRows(), T_at_ag.nRows() + T_bt_bo.nRows(), T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() };
    std::vector<std::size_t> colStarts = { 0, T_at_ag.nCols(), T_at_ag.nCols() + T_bt_bo.nCols(), T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() };

    tau.H.setBlock(rowStarts[0], colStarts[0], T_at_ag);
    tau.H.setBlock(rowStarts[0], colStarts[2], T_at_ab);
    tau.H.setBlock(rowStarts[1], colStarts[1], T_bt_bo);
    tau.H.setBlock(rowStarts[1], colStarts[2], T_bt_ba);
    tau.H.setBlock(rowStarts[2], colStarts[0], T_gt_ga);
    tau.H.setBlock(rowStarts[2], colStarts[3], T_gt_go);
    tau.H.setBlock(rowStarts[3], colStarts[1], T_ot_ob);
    tau.H.setBlock(rowStarts[3], colStarts[3], T_ot_og);

    // Compute and set T_tau
    Matrix<double> T_RHS = tau.H * tau.S;
    tau.T = T_LHS + T_RHS;

    return;

}

void FISHPACKHPSMethod::reorderOperators_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Form permutation vector and block sizes
    int nSide = alpha.grid.nPointsX();
    Vector<int> pi_noChange = {0, 1, 2, 3};
    Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
    Vector<int> blockSizes1(4, nSide);
    Vector<int> blockSizes2(8, nSide);

    // Permute S and T
    tau.S = tau.S.blockPermute(pi_noChange, pi_WESN, blockSizes1, blockSizes2);
    tau.T = tau.T.blockPermute(pi_WESN, pi_WESN, blockSizes2, blockSizes2);

    return;

}

void FISHPACKHPSMethod::mergePatch_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    FISHPACKFVGrid mergedGrid(alpha.grid.nPointsX() + beta.grid.nPointsX(), alpha.grid.nPointsY() + gamma.grid.nPointsY(), alpha.grid.xLower(), beta.grid.xUpper(), alpha.grid.yLower(), gamma.grid.yUpper());
    tau.grid = mergedGrid;
    tau.level = alpha.level - 1;
    tau.isLeaf = false;

    return;

}

void FISHPACKHPSMethod::coarsenUpwards_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Check for adaptivity
    std::vector<FISHPACKPatch*> patchPointers = {&alpha, &beta, &gamma, &omega};

    for (auto i = 0; i < 4; i++) {
        int nCoarsens = patchPointers[i]->nCoarsens;
        for (auto n = 0; n < nCoarsens; n++) {
            FISHPACKFVGrid coarseGrid = patchPointers[i]->grid;
            FISHPACKFVGrid fineGrid = FISHPACKFVGrid(coarseGrid.nPointsX()*2, coarseGrid.nPointsY()*2, coarseGrid.xLower(), coarseGrid.xUpper(), coarseGrid.yLower(), coarseGrid.yUpper());
            int nFine = fineGrid.nPointsX() * pow(2,nCoarsens-n-1);
            int nCoarse = coarseGrid.nPointsX() * pow(2,nCoarsens-n-1);
        
            InterpolationMatrixFine2Coarse<double> L21Side(nCoarse);
            std::vector<Matrix<double>> L21Diagonals = {L21Side, L21Side, L21Side, L21Side};
            Matrix<double> L21Patch = blockDiagonalMatrix(L21Diagonals);
            patchPointers[i]->h = L21Patch * patchPointers[i]->h;
        }
    }

    return;

}

void FISHPACKHPSMethod::mergeW_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Form hDiff
    Vector<double> h_ga = gamma.h(IS_gamma_alpha_);
    Vector<double> h_ag = alpha.h(IS_alpha_gamma_);
    Vector<double> h_ob = omega.h(IS_omega_beta_);
    Vector<double> h_bo = beta.h(IS_beta_omega_);
    Vector<double> h_ba = beta.h(IS_beta_alpha_);
    Vector<double> h_ab = alpha.h(IS_alpha_beta_);
    Vector<double> h_og = omega.h(IS_omega_gamma_);
    Vector<double> h_go = gamma.h(IS_gamma_omega_);

    Vector<double> hDiff_gamma_alpha = h_ga - h_ag;
    Vector<double> hDiff_omega_beta = h_ob - h_bo;
    Vector<double> hDiff_beta_alpha = h_ba - h_ab;
    Vector<double> hDiff_omega_gamma = h_og - h_go;

    Vector<double> hDiff = concatenate({
        hDiff_gamma_alpha,
        hDiff_omega_beta,
        hDiff_beta_alpha,
        hDiff_omega_gamma
    });

    // Compute and set w_tau
    tau.w = solve(tau.X, hDiff);

    return;

}

void FISHPACKHPSMethod::mergeH_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Compute and set h_tau
    tau.h = tau.H * tau.w;

    // Update with boundary h
    Vector<double> h_alpha_tau = alpha.h(IS_alpha_tau_);
    Vector<double> h_beta_tau = beta.h(IS_beta_tau_);
    Vector<double> h_gamma_tau = gamma.h(IS_gamma_tau_);
    Vector<double> h_omega_tau = omega.h(IS_omega_tau_);
    Vector<double> hUpdate = concatenate({
        h_alpha_tau,
        h_beta_tau,
        h_gamma_tau,
        h_omega_tau
    });
    tau.h += hUpdate;

    return;
    
}

void FISHPACKHPSMethod::reorderOperatorsUpwards_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    int nAlpha = alpha.grid.nPointsX();
    int nBeta = beta.grid.nPointsX();
    int nGamma = gamma.grid.nPointsX();
    int nOmega = omega.grid.nPointsX();
    Vector<int> n = {nAlpha, nBeta, nGamma, nOmega};
    if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
        throw std::invalid_argument("[EllipticForest::FISHPACK::FISHPACKHPSMethod::createIndexSets_] Size of children patches are not the same; something probably went wrong with the coarsening...");
    }

    // Form permutation vector and block sizes
    int nSide = alpha.grid.nPointsX();
    Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
    Vector<int> blockSizes(8, nSide);

    // Reorder
    tau.h = tau.h.blockPermute(pi_WESN, blockSizes);

}

void FISHPACKHPSMethod::uncoarsen_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    EllipticForestApp& app = EllipticForestApp::getInstance();

    for (auto n = 0; n < tau.nCoarsens; n++) {
        FISHPACKFVGrid coarseGrid = tau.grid;
        FISHPACKFVGrid fineGrid = FISHPACKFVGrid(coarseGrid.nPointsX()*2, coarseGrid.nPointsY()*2, coarseGrid.xLower(), coarseGrid.xUpper(), coarseGrid.yLower(), coarseGrid.yUpper());

        int nFine = fineGrid.nPointsX();
        int nCoarse = coarseGrid.nPointsX();

        InterpolationMatrixCoarse2Fine<double> L12Side(nFine);
        std::vector<Matrix<double>> L12Diagonals = {L12Side, L12Side, L12Side, L12Side};
        Matrix<double> L12Patch = blockDiagonalMatrix(L12Diagonals);
        tau.g = L12Patch * tau.g;

        tau.grid = fineGrid;
    }

    return;

}

void FISHPACKHPSMethod::applyS_(FISHPACKPatch& tau, FISHPACKPatch& alpha, FISHPACKPatch& beta, FISHPACKPatch& gamma, FISHPACKPatch& omega) {

    // Apply solution operator to get interior of tau
    Vector<double> u_tau_interior = tau.S * tau.g;

    // Apply non-homogeneous contribution
    EllipticForestApp& app = EllipticForestApp::getInstance();
    if (!std::get<bool>(app.options["homogeneous-rhs"])) {
        u_tau_interior = u_tau_interior + tau.w;
    }

    // Extract components of interior of tau
    int nSide = alpha.grid.nPointsX();
    Vector<double> g_alpha_gamma = u_tau_interior.getSegment(0*nSide, nSide);
    Vector<double> g_beta_omega = u_tau_interior.getSegment(1*nSide, nSide);
    Vector<double> g_alpha_beta = u_tau_interior.getSegment(2*nSide, nSide);
    Vector<double> g_gamma_omega = u_tau_interior.getSegment(3*nSide, nSide);

    // Extract components of exterior of tau
    Vector<double> g_alpha_W = tau.g.getSegment(0*nSide, nSide);
    Vector<double> g_gamma_W = tau.g.getSegment(1*nSide, nSide);
    Vector<double> g_beta_E = tau.g.getSegment(2*nSide, nSide);
    Vector<double> g_omega_E = tau.g.getSegment(3*nSide, nSide);
    Vector<double> g_alpha_S = tau.g.getSegment(4*nSide, nSide);
    Vector<double> g_beta_S = tau.g.getSegment(5*nSide, nSide);
    Vector<double> g_gamma_N = tau.g.getSegment(6*nSide, nSide);
    Vector<double> g_omega_N = tau.g.getSegment(7*nSide, nSide);

    // Set child patch Dirichlet data
    alpha.g = concatenate({g_alpha_W, g_alpha_beta, g_alpha_S, g_alpha_gamma});
    beta.g = concatenate({g_alpha_beta, g_beta_E, g_beta_S, g_beta_omega});
    gamma.g = concatenate({g_gamma_W, g_gamma_omega, g_alpha_gamma, g_gamma_N});
    omega.g = concatenate({g_gamma_omega, g_omega_E, g_beta_omega, g_omega_N});

}
#endif

} // NAMESPACE : FISHPACK

} // NAMESPACE : EllipticSolver