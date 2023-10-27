#include <Vector.hpp>
#include <Matrix.hpp>
#include <random>

template<typename T>
T randomRange(T lower, T upper) {
    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_real_distribution<T> distribution(lower, upper);
    return distribution(engine);
}

using namespace EllipticForest;

int rank, size;
int PROBLEM_SIZE;

void TEST_petsc_mat_vec_multiply() {

    // Set size of problem
    int M = PROBLEM_SIZE;
    int N = PROBLEM_SIZE;

    // Create parallel matrix
    ParallelMatrix<double> A(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, MATMPIDENSE);

    // Get ownership range
    int i_local_first, i_local_last;
    MatGetOwnershipRange(A.mat, &i_local_first, &i_local_last);

    // Set local entries to random values
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<int> col_indices = vectorRange(0, N-1);
        Matrix<double> local_values(row_indices.size(), col_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            for (int j = 0; j < col_indices.size(); j++) {
                local_values(i,j) = randomRange<double>(-1., 1.);
            }
        }
        A.setValues(row_indices, col_indices, local_values, INSERT_VALUES);
        A.beginAssembly(MAT_FINAL_ASSEMBLY);
        A.endAssembly(MAT_FINAL_ASSEMBLY);
    }

    // View matrix
    if (M <= 8 && N <= 8) MatView(A.mat, 0);

    // Create parallel vector
    ParallelVector<double> x(MPI_COMM_WORLD, PETSC_DECIDE, N);

    // Get ownership range
    VecGetOwnershipRange(x.vec, &i_local_first, &i_local_last);

    // Set local entries to random values
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<double> local_values(row_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            local_values[i] = randomRange<double>(-1., 1.);
        }
        x.setValues(row_indices, local_values, INSERT_VALUES);
        x.beginAssembly();
        x.endAssembly();
    }

    // View vector
    if (M <= 8 && N <= 8) VecView(x.vec, 0);

    // Create parallel vector (output)
    ParallelVector<double> y(MPI_COMM_WORLD, PETSC_DECIDE, N);

    // Perform matrix-vector multiplication
    double t_start, t_final;
    t_start = MPI_Wtime();
    MatMult(A.mat, x.vec, y.vec);
    t_final = MPI_Wtime();
    double t_elapsed = t_final - t_start;
    MPI_Reduce(&t_elapsed, &t_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t_elapsed = t_elapsed / size;
    if (rank == 0)
        printf("[RANK %i/%i] mat-vec mult time  = %f [sec]\n", rank, size, t_elapsed);

    // View vector
    if (M <= 8 && N <= 8) VecView(y.vec, 0);

}

void TEST_petsc_mat_vec_solve() {

    // Set size of problem
    int M = PROBLEM_SIZE;
    int N = PROBLEM_SIZE;

    // Create parallel matrix
    ParallelMatrix<double> A(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N);

    // Get ownership range
    int i_local_first, i_local_last;
    MatGetOwnershipRange(A.mat, &i_local_first, &i_local_last);

    // Set local entries to random values
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<int> col_indices = vectorRange(0, N-1);
        Matrix<double> local_values(row_indices.size(), col_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            for (int j = 0; j < col_indices.size(); j++) {
                local_values(i,j) = randomRange<double>(-1., 1.);
            }
        }
        A.setValues(row_indices, col_indices, local_values, INSERT_VALUES);
        A.beginAssembly(MAT_FINAL_ASSEMBLY);
        A.endAssembly(MAT_FINAL_ASSEMBLY);
    }

    // View matrix
    if (M <= 8 && N <= 8) MatView(A.mat, 0);

    // Create parallel vector
    ParallelVector<double> b(MPI_COMM_WORLD, PETSC_DECIDE, N);

    // Get ownership range
    VecGetOwnershipRange(b.vec, &i_local_first, &i_local_last);

    // Set local entries to random values
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<double> local_values(row_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            local_values[i] = randomRange<double>(-1., 1.);
        }
        b.setValues(row_indices, local_values, INSERT_VALUES);
        b.beginAssembly();
        b.endAssembly();
    }

    // View vector
    if (M <= 8 && N <= 8) VecView(b.vec, 0);

    // Create parallel vector (output)
    ParallelVector<double> x(MPI_COMM_WORLD, PETSC_DECIDE, N);
    
    // Setup factorization
    IS row_perm, col_perm;
    Vector<int> idx_row = vectorRange(0, M-1);
    Vector<int> idx_col = vectorRange(0, N-1);
    ISCreateGeneral(MPI_COMM_WORLD, M, idx_row.data().data(), PETSC_COPY_VALUES, &row_perm);
    ISCreateGeneral(MPI_COMM_WORLD, N, idx_col.data().data(), PETSC_COPY_VALUES, &col_perm);
    MatFactorInfo mat_factor_info;
    MatFactorInfoInitialize(&mat_factor_info);

    // Solve linear system
    double t_start, t_final;
    t_start = MPI_Wtime();
    MatLUFactor(A.mat, row_perm, col_perm, &mat_factor_info);
    MatSolve(A.mat, b.vec, x.vec);
    t_final = MPI_Wtime();
    double t_elapsed = t_final - t_start;
    MPI_Reduce(&t_elapsed, &t_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t_elapsed = t_elapsed / size;
    if (rank == 0)
        printf("[RANK %i/%i] mat-vec solve time = %f [sec]\n", rank, size, t_elapsed);

    // View vector
    if (M <= 8 && N <= 8) VecView(x.vec, 0);

}

void TEST_petsc_mat_mat_solve() {

    // Set size of problem
    int M = PROBLEM_SIZE;
    int N = PROBLEM_SIZE;

    // Create parallel matrix
    ParallelMatrix<double> A(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, MATSCALAPACK);
    ParallelMatrix<double> B(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, MATSCALAPACK);
    ParallelMatrix<double> X(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, MATSCALAPACK);

    // Get ownership range
    int i_local_first, i_local_last;
    MatGetOwnershipRange(A.mat, &i_local_first, &i_local_last);

    // Set local entries to random values
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<int> col_indices = vectorRange(0, N-1);
        Matrix<double> local_values(row_indices.size(), col_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            for (int j = 0; j < col_indices.size(); j++) {
                local_values(i,j) = randomRange<double>(-1., 1.);
            }
        }
        A.setValues(row_indices, col_indices, local_values, INSERT_VALUES);
        A.beginAssembly(MAT_FINAL_ASSEMBLY);
    }
    {
        Vector<int> row_indices = vectorRange(i_local_first, i_local_last-1);
        Vector<int> col_indices = vectorRange(0, N-1);
        Matrix<double> local_values(row_indices.size(), col_indices.size());
        for (int i = 0; i < row_indices.size(); i++) {
            for (int j = 0; j < col_indices.size(); j++) {
                local_values(i,j) = randomRange<double>(-1., 1.);
            }
        }
        B.setValues(row_indices, col_indices, local_values, INSERT_VALUES);
        B.beginAssembly(MAT_FINAL_ASSEMBLY);
    }
    
    // Finalize assembly
    A.endAssembly(MAT_FINAL_ASSEMBLY);
    B.endAssembly(MAT_FINAL_ASSEMBLY);
    X.beginAssembly(MAT_FINAL_ASSEMBLY);
    X.endAssembly(MAT_FINAL_ASSEMBLY);

    // View matrix
    if (M <= 8 && N <= 8) MatView(A.mat, 0);
    if (M <= 8 && N <= 8) MatView(B.mat, 0);

    // Setup factorization
    IS row_perm, col_perm;
    Vector<int> idx_row = vectorRange(0, M-1);
    Vector<int> idx_col = vectorRange(0, N-1);
    ISCreateGeneral(MPI_COMM_WORLD, M, idx_row.data().data(), PETSC_COPY_VALUES, &row_perm);
    ISCreateGeneral(MPI_COMM_WORLD, N, idx_col.data().data(), PETSC_COPY_VALUES, &col_perm);
    MatFactorInfo mat_factor_info;
    MatFactorInfoInitialize(&mat_factor_info);
    
    // Solve linear system
    double t_start, t_final;
    t_start = MPI_Wtime();
    MatLUFactor(A.mat, row_perm, col_perm, &mat_factor_info);
    MatMatSolve(A.mat, B.mat, X.mat);
    t_final = MPI_Wtime();
    double t_elapsed = t_final - t_start;
    MPI_Reduce(&t_elapsed, &t_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t_elapsed = t_elapsed / size;
    if (rank == 0)
        printf("[RANK %i/%i] mat-mat solve time = %f [sec]\n", rank, size, t_elapsed);

    // View vector
    if (M <= 8 && N <= 8) MatView(X.mat, 0);

}

int main(int argc, char** argv) {

    MPI_Init(nullptr, nullptr);
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    if (argc > 1)
        PROBLEM_SIZE = atoi(argv[1]);
    else
        PROBLEM_SIZE = 8;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_mat_vec_multiply();

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_mat_vec_solve();

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_mat_mat_solve();

    PetscFinalize();
    MPI_Finalize();

}