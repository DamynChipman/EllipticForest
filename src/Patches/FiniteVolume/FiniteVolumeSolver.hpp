#ifndef FINITE_VOLUME_SOLVER_HPP_
#define FINITE_VOLUME_SOLVER_HPP_

#include "FiniteVolumeGrid.hpp"
#include "../../Vector.hpp"
#include "../../Matrix.hpp"
#include "../../PatchSolver.hpp"

namespace EllipticForest {

using Analytical2DFunction = std::function<double(double, double)>;

enum FiniteVolumeSolverType {
    FivePointStencil,
    FISHPACK90
};

extern "C" {
	void hstcrt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR);
	void hstcrtt_(double* A, double* B, int* M, int* MBDCND, double* BDA, double* BDB, double* C, double* D, int* N, int* NBDCND, double* BDC, double* BDD, double* ELMBDA, double* F, int* IDIMF, double* PERTRB, int* IERROR, double* W);
}

namespace Petsc {
    using KSP = KSP;
    using PC = PC;
} // NAMESPACE : Petsc

class FiniteVolumeSolver : public MPI::MPIObject {

public:

    Petsc::KSP ksp;
    Petsc::PC pc;

    FiniteVolumeSolverType solver_type = FiniteVolumeSolverType::FivePointStencil;

    Analytical2DFunction alpha_function;
    Analytical2DFunction beta_function;
    Analytical2DFunction lambda_function;
    Analytical2DFunction rhs_function;

    FiniteVolumeSolver();
    FiniteVolumeSolver(MPI::Communicator comm, Analytical2DFunction alpha_function, Analytical2DFunction beta_function, Analytical2DFunction lambda_function, Analytical2DFunction rhs_function);

    std::string name();
    Vector<double> solve(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data);
    Vector<double> mapD2N(FiniteVolumeGrid& grid, Vector<double>& dirichlet_data, Vector<double>& rhs_data);
    Matrix<double> buildD2N(FiniteVolumeGrid& grid);
    Vector<double> particularNeumannData(FiniteVolumeGrid& grid, Vector<double>& rhs_data);
    int gridIndex2MatrixIndex(int i, int j, int nx, int ny);

};

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_SOLVER_HPP_