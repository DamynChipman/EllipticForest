#include "FiniteVolumeSolver.hpp"

namespace EllipticForest {

FiniteVolumeSolver::FiniteVolumeSolver() :
    MPIObject(MPI_COMM_SELF)
        {}

FiniteVolumeSolver::FiniteVolumeSolver(MPI::Communicator comm, Analytical2DFunction alpha_function, Analytical2DFunction beta_function, Analytical2DFunction lambda_function, Analytical2DFunction rhs_function) :
    MPIObject(comm),
    alpha_function(alpha_function),
    beta_function(beta_function),
    lambda_function(lambda_function),
    rhs_function(rhs_function) {


    //

}

int FiniteVolumeSolver::gridIndex2MatrixIndex(int i, int j, int nx, int ny) {
    return j + i*ny;
}

std::string FiniteVolumeSolver::name() {
    return "FiniteVolumeSolver";
}

Vector<double> FiniteVolumeSolver::solve(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data) {

    if (solver_type == FiniteVolumeSolverType::FivePointStencil) {
        // Unpack Dirichlet data
        int nSide = grid.nx();
        Vector<double> gWest = dirichlet_data.getSegment(0*nSide, nSide);
        Vector<double> gEast = dirichlet_data.getSegment(1*nSide, nSide);
        Vector<double> gSouth = dirichlet_data.getSegment(2*nSide, nSide);
        Vector<double> gNorth = dirichlet_data.getSegment(3*nSide, nSide);

        // Unpack grid data
        int nx = grid.nx();
        int ny = grid.ny();
        int N = nx*ny;
        double dx = grid.dx();
        double dy = grid.dy();

        // Get Petsc data
        // ParallelMatrix<double> A(MPI_COMM_SELF, N, N, N, N, MATDENSE);
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

                double alpha_ij = alpha_function(xi, yj);
                double beta_ipj = beta_function(xip, yj);
                double beta_imj = beta_function(xim, yj);
                double beta_ijp = beta_function(xi, yjp);
                double beta_ijm = beta_function(xi, yjm);
                double lambda_ij = lambda_function(xi, yj);

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

                double f_boundary = rhs_data[IC];
                VecSetValue(f, IC, rhs_data[IC], INSERT_VALUES);

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
        // KSP ksp;
        // PC pc;
        // KSPCreate(MPI_COMM_SELF, &ksp);
        // // KSPSetType(ksp, KSPPREONLY);
        // // KSPGetPC(ksp, &pc);
        // // PCSetType(pc, PCLU);
        // KSPSetOperators(ksp, A, A);
        // KSPSolve(ksp, f, x);
        MatFactorInfo mat_factor_info;
        MatFactorInfoInitialize(&mat_factor_info);
        IS row_perm, col_perm;
        ISCreateStride(MPI_COMM_SELF, N, 0, 1, &row_perm);
        ISCreateStride(MPI_COMM_SELF, N, 0, 1, &col_perm);
        MatLUFactor(A, row_perm, col_perm, &mat_factor_info);
        MatSolve(A, f, x);

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
        // KSPDestroy(&ksp);

        return x_vector;
    }
    else if (solver_type == FiniteVolumeSolverType::FISHPACK90) {
        
        // Unpack Dirichlet Data
        int nSide = grid.nx();
        Vector<double> gWest = dirichlet_data.getSegment(0*nSide, nSide);
        Vector<double> gEast = dirichlet_data.getSegment(1*nSide, nSide);
        Vector<double> gSouth = dirichlet_data.getSegment(2*nSide, nSide);
        Vector<double> gNorth = dirichlet_data.getSegment(3*nSide, nSide);

        // Transpose RHS data for FORTRAN call
        Vector<double> fT(nSide * nSide);
        for (int i = 0; i < grid.nx(); i++) {
            for (int j = 0; j < grid.ny(); j++) {
                fT[i + j*nSide] = rhs_data[j + i*nSide];
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
        double ELMBDA = lambda_function(0,0); // @TODO: Implement or get lambda value
        double* F = fT.dataPointer();
        // double* F = rhs_data.dataPointer();
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

        return solution; // return rhs_data;

    }

}

Vector<double> FiniteVolumeSolver::mapD2N(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data) {

    // Unpack grid data
	int nSide = grid.nx();

	// Unpack Dirichlet data
	Vector<double> gWest = dirichlet_data.getSegment(0*nSide, nSide);
	Vector<double> gEast = dirichlet_data.getSegment(1*nSide, nSide);
	Vector<double> gSouth = dirichlet_data.getSegment(2*nSide, nSide);
	Vector<double> gNorth = dirichlet_data.getSegment(3*nSide, nSide);

	// Compute solution on interior nodes
	Vector<double> u = solve(grid, dirichlet_data, rhs_data);

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

Matrix<double> FiniteVolumeSolver::buildD2N(FiniteVolumeGrid& grid) {

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

Vector<double> FiniteVolumeSolver::particularNeumannData(FiniteVolumeGrid& grid, Vector<double>& rhs_data) {

    Vector<double> gZero(2*grid.nx() + 2*grid.ny(), 0);
    return mapD2N(grid, gZero, rhs_data);

}

} // NAMESPACE : EllipticForest