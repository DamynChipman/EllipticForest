#include "PETSc.hpp"

namespace EllipticForest {

namespace Petsc {

PetscGrid::PetscGrid() {}

PetscGrid::PetscGrid(int nx, int ny, double xLower, double xUpper, double yLower, double yUpper) :
    nx_(nx), ny_(ny), xLower_(xLower), xUpper_(xUpper), yLower_(yLower), yUpper_(yUpper) {

    dx_ = (xUpper_ - xLower_) / (nx_);
    dy_ = (yUpper_ - yLower_) / (ny_); // AHHHHHHHH!!!! This was yUpper_ - xUpper_... Three days lost to debugging...

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

double PetscGrid::dy() { return dy_; }

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

PetscPatchSolver::PetscPatchSolver() :
    MPIObject(MPI_COMM_WORLD)
        {}

PetscPatchSolver::PetscPatchSolver(MPI_Comm comm) :
    MPIObject(comm)
        {}

std::string PetscPatchSolver::name() { return "PETScPatchSolver"; }

Vector<double> PetscPatchSolver::solve(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

    // std::this_thread::sleep_for(std::chrono::seconds(this->getRank()));
    // Unpack Dirichlet data
    int nSide = grid.nPointsX();
    Vector<double> gWest = dirichletData.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichletData.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichletData.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichletData.getSegment(3*nSide, nSide);

    // Unpack grid data
    int nx = grid.nPointsX();
    int ny = grid.nPointsY();
    int N = nx*ny;
    double dx = grid.dx();
    double dy = grid.dy();

    // Get Petsc data
    Mat A;
    MatCreate(MPI_COMM_SELF, &A);
    MatSetSizes(A, N, N, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);

    Vec f;
    VecCreate(MPI_COMM_SELF, &f);
    VecSetSizes(f, N, N);
    VecSetFromOptions(f);

    Vec x;
    VecDuplicate(f, &x);

    PetscInt m, n;
    PetscInt idxm[1], idxn[5];
    PetscScalar v[5];

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            double xi = grid(0, i);
            double yj = grid(1, j);

            double xip = xi + dx/2.0;
            double xim = xi - dx/2.0;
            double yjp = yj + dy/2.0;
            double yjm = yj - dy/2.0;

            double alpha_ij = alphaFunction(xi, yj);
            double beta_ipj = betaFunction(xip, yj);
            double beta_imj = betaFunction(xim, yj);
            double beta_ijp = betaFunction(xi, yjp);
            double beta_ijm = betaFunction(xi, yjm);
            double lambda_ij = lambdaFunction(xi, yj);

            double cW = alpha_ij * beta_imj / pow(dx, 2);
            double cE = alpha_ij * beta_ipj / pow(dx, 2);
            double cS = alpha_ij * beta_ijm / pow(dy, 2);
            double cN = alpha_ij * beta_ijp / pow(dy, 2);
            double cC = -alpha_ij * ((beta_ipj + beta_imj)/pow(dx, 2) + (beta_ijp + beta_ijm)/pow(dy, 2)) + lambda_ij;

            int IC = gridIndex2MatrixIndex(i, j, nx, ny);
            int JW = gridIndex2MatrixIndex(i-1, j, nx, ny);
            int JE = gridIndex2MatrixIndex(i+1, j, nx, ny);
            int JS = gridIndex2MatrixIndex(i, j-1, nx, ny);
            int JN = gridIndex2MatrixIndex(i, j+1, nx, ny);
            int JC = gridIndex2MatrixIndex(i, j, nx, ny);
            
            m = 1;
            idxm[0] = IC;

            double f_boundary = rhsData[IC];
            VecSetValue(f, IC, rhsData[IC], INSERT_VALUES);

            if (i == 0 && j == 0) {
                // Lower-left corner
                n = 3;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JN; v[1] = cN;
                idxn[2] = JC; v[2] = cC - cW - cS;
                f_boundary = -2.0*(cW*gWest[j] + cS*gSouth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (i == nx-1 && j == 0) {
                // Lower-right corner
                n = 3;
                idxn[0] = JW; v[0] = cW;
                idxn[1] = JN; v[1] = cN;
                idxn[2] = JC; v[2] = cC - cE - cS;
                f_boundary = -2.0*(cE*gEast[j] + cS*gSouth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (i == 0 && j == ny-1) {
                // Upper-left corner
                n = 3;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JS; v[1] = cS;
                idxn[2] = JC; v[2] = cC - cW - cN;
                f_boundary = -2.0*(cW*gWest[j] + cN*gNorth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (i == nx-1 && j == ny-1) {
                // Upper-right corner
                n = 3;
                idxn[0] = JW; v[0] = cW;
                idxn[1] = JS; v[1] = cS;
                idxn[2] = JC; v[2] = cC - cE - cN;
                f_boundary = -2.0*(cE*gEast[j] + cN*gNorth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (i == 0) {
                // West edge
                n = 4;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JN; v[1] = cN;
                idxn[2] = JS; v[2] = cS;
                idxn[3] = JC; v[3] = cC - cW;
                f_boundary = -2.0*(cW*gWest[j]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (i == nx-1) {
                // East edge
                n = 4;
                idxn[0] = JW; v[0] = cW;
                idxn[1] = JN; v[1] = cN;
                idxn[2] = JS; v[2] = cS;
                idxn[3] = JC; v[3] = cC - cE;
                f_boundary = -2.0*(cE*gEast[j]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (j == 0) {
                // South edge
                n = 4;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JW; v[1] = cW;
                idxn[2] = JN; v[2] = cN;
                idxn[3] = JC; v[3] = cC - cS;
                f_boundary = -2.0*(cS*gSouth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else if (j == ny-1) {
                // North edge
                n = 4;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JW; v[1] = cW;
                idxn[2] = JS; v[2] = cS;
                idxn[3] = JC; v[3] = cC - cN;
                f_boundary = -2.0*(cN*gNorth[i]);
                VecSetValue(f, IC, f_boundary, ADD_VALUES);
            }
            else {
                // Interior
                n = 5;
                idxn[0] = JE; v[0] = cE;
                idxn[1] = JW; v[1] = cW;
                idxn[2] = JN; v[2] = cN;
                idxn[3] = JS; v[3] = cS;
                idxn[4] = JC; v[4] = cC;
            }
            MatSetValues(A, m, idxm, n, idxn, v, INSERT_VALUES);

        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(f);

    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(f);

    // Solve the linear system
    KSP ksp;
    PC pc;
    KSPCreate(MPI_COMM_SELF, &ksp);
    // KSPSetType(ksp, KSPPREONLY);
    // KSPGetPC(ksp, &pc);
    // PCSetType(pc, PCLU);
    KSPSetOperators(ksp, A, A);
    KSPSolve(ksp, f, x);

    // Create EllipticForest vector
    double* x_data;
    VecGetArray(x, &x_data);
    std::vector<double> xv;
    xv.assign(x_data, x_data + N);
    Vector<double> x_vector(xv);

    // Clean up
    MatDestroy(&A);
    VecDestroy(&x);
    VecDestroy(&f);
    KSPDestroy(&ksp);

    return x_vector;

}

Vector<double> PetscPatchSolver::mapD2N(PetscGrid& grid, Vector<double>& dirichletData, Vector<double>& rhsData) {

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

Matrix<double> PetscPatchSolver::buildD2N(PetscGrid& grid) {

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

Vector<double> PetscPatchSolver::particularNeumannData(PetscGrid& grid, Vector<double>& rhsData) {
    Vector<double> gZero(2*grid.nPointsX() + 2*grid.nPointsY(), 0);
    return mapD2N(grid, gZero, rhsData);
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

void PetscPatchSolver::setRHSFunction(std::function<double(double, double)> fn) {
    rhsFunction = fn;
}

int PetscPatchSolver::gridIndex2MatrixIndex(int i, int j, int nx, int ny) {
    return j + i*ny;
}

PetscPatch::PetscPatch() {}

PetscPatch::PetscPatch(PetscGrid grid) :
    grid_(grid)
        {}

std::string PetscPatch::name() {
    return "PetscPatch";
}

PetscGrid& PetscPatch::grid() {
    return grid_;
}

PetscPatch PetscPatch::buildChild(std::size_t childIndex) {
    return {};   
}

double PetscPatch::dataSize() {
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

std::string PetscPatch::str() {
    std::string res;

    // res += "globalID = " + std::to_string(globalID) + "\n";
    // res += "leafID = " + std::to_string(leafID) + "\n";
    // res += "level = " + std::to_string(level) + "\n";
    // res += "isLeaf = " + std::to_string(isLeaf) + "\n";
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

PetscPatchNodeFactory::PetscPatchNodeFactory() {}

PetscPatchNodeFactory::PetscPatchNodeFactory(MPI_Comm comm) :
    MPIObject(comm)
        {}

Node<PetscPatch> PetscPatchNodeFactory::createNode(PetscPatch data, std::string path, int level, int pfirst, int plast) {
    Node<PetscPatch> node(this->getComm(), data, path, level, pfirst, plast);
    return node;
}

Node<PetscPatch> PetscPatchNodeFactory::createChildNode(Node<PetscPatch> parentNode, int siblingID, int pfirst, int plast) {
    
    // Get parent grid info
    auto& parentGrid = parentNode.data.grid();
    int nx = parentGrid.nPointsX();
    int ny = parentGrid.nPointsY();
    double xLower = parentGrid.xLower();
    double xUpper = parentGrid.xUpper();
    double xMid = (xLower + xUpper) / 2.0;
    double yLower = parentGrid.yLower();
    double yUpper = parentGrid.yUpper();
    double yMid = (yLower + yUpper) / 2.0;

    // Create grid communicator
    // MPI_Group gridGroup, parentGroup;
    // MPI_Comm gridComm;
    

    // Create child grid
    PetscGrid childGrid;
    switch (siblingID) {
        case 0:
            // Lower left
            childGrid = PetscGrid(nx, ny, xLower, xMid, yLower, yMid);
            break;
        case 1:
            // Lower right
            childGrid = PetscGrid(nx, ny, xMid, xUpper, yLower, yMid);
            break;
        case 2:
            // Upper left
            childGrid = PetscGrid(nx, ny, xLower, xMid, yMid, yUpper);
            break;
        case 3:
            // Upper right
            childGrid = PetscGrid(nx, ny, xMid, xUpper, yMid, yUpper);
            break;
        default:
            break;
    }

    // Create communicator for child patch
    // MPI::Group child_group;
    // MPI::Communicator child_comm;
    // parentNode.getMPIGroupComm(&child_group, &child_comm);

    // Create child patch
    PetscPatch childPatch(childGrid);

    // Create child node
    std::string path = parentNode.path + std::to_string(siblingID);
    int level = parentNode.level + 1;
    return Node<PetscPatch>(this->getComm(), childPatch, path, level, pfirst, plast);
    
}

Node<PetscPatch> PetscPatchNodeFactory::createParentNode(std::vector<Node<PetscPatch>> childNodes, int pfirst, int plast) {

    // Create parent grid
    int nx = childNodes[0].data.grid().nPointsX();
    int ny = childNodes[0].data.grid().nPointsY();
    double xLower = childNodes[0].data.grid().xLower();
    double xUpper = childNodes[1].data.grid().xUpper();
    double yLower = childNodes[0].data.grid().yLower();
    double yUpper = childNodes[2].data.grid().yUpper();
    PetscGrid parentGrid(nx, ny, xLower, xUpper, yLower, yUpper);

    // Create communicator for parent patch
    // MPI::Group alpha_beta_group;
    // MPI::Group gamma_omega_group;
    

    // Create parent patch
    PetscPatch parentPatch(parentGrid); // TODO: Switch MPI_COMM_WORLD to patch communicator

    // Create parent node
    std::string path = childNodes[0].path.substr(0, childNodes[0].path.length()-1);
    int level = childNodes[0].level - 1;
    return Node<PetscPatch>(this->getComm(), parentPatch, path, level, pfirst, plast);

}

} // NAMESPACE : Petsc

namespace MPI {

template<>
int broadcast(Petsc::PetscGrid& grid, int root, MPI_Comm comm) {
    int nx = grid.nPointsX();
    int ny = grid.nPointsY();
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
    if (rank!= root) grid = Petsc::PetscGrid(nx, ny, xLower, xUpper, yLower, yUpper);
    return 1;
} 

template<>
int broadcast(Petsc::PetscPatch& patch, int root, MPI_Comm comm) {
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

} // NAMESPACE : EllipticForest