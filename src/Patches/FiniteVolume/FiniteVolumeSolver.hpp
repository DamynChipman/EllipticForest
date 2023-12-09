#ifndef FINITE_VOLUME_SOLVER_HPP_
#define FINITE_VOLUME_SOLVER_HPP_

#include "FiniteVolumeGrid.hpp"
#include "../../Vector.hpp"
#include "../../Matrix.hpp"
#include "../../PatchSolver.hpp"

namespace EllipticForest {

using Analytical2DFunction = std::function<double(double, double)>;

/**
 * @brief Flags for type of finite volume solver
 * 
 */
enum FiniteVolumeSolverType {
    FivePointStencil,
    FISHPACK90
};

extern "C" {
    /**
     * @brief FISHPACK90 routine for solving the Helmholtz equation with a five-point stencil with a staggered grid
     * 
     * @param A The lower range of X
     * @param B The upper range of X
     * @param M The number of grid points in the interval (A,B)
     * @param MBDCND Type of boundary condition at X=A and X=B
     * @param BDA Double array of boundary data at X=A
     * @param BDB Double array of boundary data at X=B
     * @param C The lower range of Y
     * @param D The upper range of Y
     * @param N The number of grid points in the interval (C,D)
     * @param NBDCND Type of boundary condition at Y=C and Y=D
     * @param BDC Double array of boundary data at Y=C
     * @param BDD Double array of boundary data at Y=D
     * @param ELMBDA Scalar value of lambda in Helmholtz equation
     * @param F Logically 2D double array of RHS values; on output: Contains the solution U
     * @param IDIMF Dimension of the F array
     * @param PERTRB Perturbation constant for non-existant solution
     * @param IERROR Error flag
     */
	void hstcrt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR);
	// void hstcrtt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR, double* W);
}

namespace Petsc {
    using KSP = KSP;
    using PC = PC;
} // NAMESPACE : Petsc

class FiniteVolumeSolver : public MPI::MPIObject {

public:

    // Petsc::KSP ksp;
    // Petsc::PC pc;

    /**
     * @brief Solver type
     * 
     */
    FiniteVolumeSolverType solver_type = FiniteVolumeSolverType::FivePointStencil;

    /**
     * @brief Analytical function for alpha(x,y)
     * 
     */
    Analytical2DFunction alpha_function;

    /**
     * @brief Analytical function for beta(x,y)
     * 
     */
    Analytical2DFunction beta_function;

    /**
     * @brief Analytical function for lambda(x,y)
     * 
     */
    Analytical2DFunction lambda_function;

    /**
     * @brief Analytical function the RHS of elliptic problem
     * 
     */
    Analytical2DFunction rhs_function;

    /**
     * @brief Construct a new FiniteVolumeSolver object (default)
     * 
     */
    FiniteVolumeSolver();

    /**
     * @brief Construct a new FiniteVolumeSolver object
     * 
     * @param comm MPI communicator
     * @param alpha_function Analytical alpha function
     * @param beta_function Analytical beta function
     * @param lambda_function Analytical lambda function
     * @param rhs_function Analytical RHS function
     */
    FiniteVolumeSolver(MPI::Communicator comm, Analytical2DFunction alpha_function, Analytical2DFunction beta_function, Analytical2DFunction lambda_function, Analytical2DFunction rhs_function);

    /**
     * @brief Returns the name of the solver
     * 
     * @return std::string 
     */
    std::string name();

    /**
     * @brief Solves the BVP given a grid, the Dirichlet data, and the RHS data
     * 
     * @param grid Finite volume grid
     * @param dirichlet_data Dirichlet data at the boundary (WESN ordering)
     * @param rhs_data RHS data (patch ordering)
     * @return Vector<double> 
     */
    Vector<double> solve(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data);

    /**
     * @brief Performs the action of mapping Dirichlet data to Neumann data
     * 
     * @param grid Finite volume grid
     * @param dirichlet_data Dirichlet data at the boundary (WESN) ordering
     * @param rhs_data RHS data (patch ordering)
     * @return Vector<double> 
     */
    Vector<double> mapD2N(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data);

    /**
     * @brief Builds the explicit Dirichlet-to-Neumann matrix T
     * 
     * @param grid Finite volume grid
     * @return Matrix<double> 
     */
    Matrix<double> buildD2N(FiniteVolumeGrid& grid);

    /**
     * @brief Builds the explicit particular Neumann data vector w
     * 
     * @param grid Finite volume grid
     * @param rhs_data RHS data (patch ordering)
     * @return Vector<double> 
     */
    Vector<double> particularNeumannData(FiniteVolumeGrid& grid, Vector<double>& rhs_data);

    /**
     * @brief Maps a grid index to a matrix index
     * 
     * @param i X-index
     * @param j Y-index
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @return int 
     */
    int gridIndex2MatrixIndex(int i, int j, int nx, int ny);

};

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_SOLVER_HPP_