#include <Matrix.hpp>
#include <thread>
#include <unistd.h>

using namespace EllipticForest;

int rank, size;

void TEST_petsc_matrix_init() {
    
    int M = 8;
    int N = 8;
    ParallelMatrix<double> mat(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N);

    Vector<int> row_indices = {0, 2, 4};
    Vector<int> col_indices = vectorRange(0, N-1);
    Matrix<double> values(row_indices.size(), col_indices.size());
    int v = 0;
    for (int i = 0; i < row_indices.size(); i++) {
        int ii = row_indices[i];
        for (int j = 0; j < col_indices.size(); j++) {
            int jj = col_indices[j];
            values(i, j) = (double) v;
            v++;
        }
    }

    mat.setValues(row_indices, col_indices, values, INSERT_VALUES);

    mat.beginAssembly(MAT_FINAL_ASSEMBLY);
    mat.endAssembly(MAT_FINAL_ASSEMBLY);

    MatView(mat.mat, PETSC_VIEWER_STDOUT_WORLD);

}

void TEST_petsc_matrix_serial_to_parallel() {

    int M = 4;
    int N = 4;
    Matrix<double> serial_matrix(M, N);
    int v = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            serial_matrix(i,j) = (double) v;
            v++;
        }
    }

    ParallelMatrix<double> parallel_matrix(MPI_COMM_WORLD, serial_matrix);
    parallel_matrix.beginAssembly(MAT_FINAL_ASSEMBLY);
    parallel_matrix.endAssembly(MAT_FINAL_ASSEMBLY);

    MatView(parallel_matrix.mat, 0);

}

void TEST_petsc_matrix_comm() {

    // Create communicators with single process each
    MPI::Communicator rank_comm;
    MPI::Group world_group, rank_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int ranges[1][3] = {rank, rank, 1};
    MPI_Group_range_incl(world_group, 1, ranges, &rank_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, rank_group, 0, &rank_comm);

    // Create local matrices on local communicator
    int M_local = 4;
    int N_local = 4;
    ParallelMatrix<double> mat_local(rank_comm, PETSC_DECIDE, PETSC_DECIDE, M_local, N_local);

    // Set values in local matrix
    Vector<int> row_indices = vectorRange(0, M_local-1);
    Vector<int> col_indices = vectorRange(0, N_local-1);
    Matrix<double> values(row_indices.size(), col_indices.size());
    int v = M_local*N_local*rank;
    for (int i = 0; i < row_indices.size(); i++) {
        int ii = row_indices[i];
        for (int j = 0; j < col_indices.size(); j++) {
            int jj = col_indices[j];
            values(i, j) = (double) v;
            v++;
        }
    }
    mat_local.setValues(row_indices, col_indices, values, INSERT_VALUES);
    mat_local.beginAssembly(MAT_FINAL_ASSEMBLY);
    mat_local.endAssembly(MAT_FINAL_ASSEMBLY);

    // Create local matrices on global communicator
    ParallelMatrix<double> mat_global(MPI_COMM_WORLD, mat_local);

    // View each local mat on global communicator (sleep for `rank` seconds so output is ordered)
    sleep(rank);
    MatView(mat_global.mat, 0);

    // Create merged mat on global communicator
    //    For this test, I just put the four locally computed matrices on the diagonal of the merged matrix
    //    In the 4-to-1 merge, this would compute T_merged from T_alpha, T_beta, T_gamma, and T_omega (children)
    int M_merged = M_local*size;
    int N_merged = N_local*size;
    ParallelMatrix<double> mat_merged(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M_merged, N_merged);

    // Get values of local matrix to put on diagonal
    Matrix<double> values_rank(row_indices.size(), col_indices.size());
    mat_global.getValues(row_indices, col_indices, values_rank);

    // Put local matrix contributions into merged matrix (placeholder for computing merged matrix)
    Vector<int> row_indices_merged = vectorRange(M_local*rank, M_local*(rank+1)-1);
    Vector<int> col_indices_merged = vectorRange(N_local*rank, N_local*(rank+1)-1);
    mat_merged.setValues(row_indices_merged, col_indices_merged, values_rank, INSERT_VALUES);
    mat_merged.beginAssembly(MAT_FINAL_ASSEMBLY);
    mat_merged.endAssembly(MAT_FINAL_ASSEMBLY);

    // View merged mat on global communicator
    sleep(rank);
    MatView(mat_merged.mat, 0);

    // Clean up
    MPI_Group_free(&world_group);
    MPI_Group_free(&rank_group);
    MPI_Comm_free(&rank_comm);

}

#if 0
void TEST_petsc_matrix_comm() {

    // Create local matrices on local communicator
    int M_local = 4;
    int N_local = 4;
    Mat mat_local;
    MatCreate(MPI_COMM_SELF, &mat_local); // Note the MPI_COMM_SELF as a substitute for a sub-communicator of MPI_COMM_WORLD
    MatSetSizes(mat_local, PETSC_DECIDE, PETSC_DECIDE, M_local, N_local);
    MatSetFromOptions(mat_local);

    // Set values in local matrix
    int* row_indices = (int*) malloc(M_local*sizeof(int));
    for (int i = 0; i < M_local; i++) {
        row_indices[i] = i;
    }

    int* col_indices = (int*) malloc(N_local*sizeof(int));
    for (int j = 0; j < M_local; j++) {;
        col_indices[j] = j;
    }

    double* values = (double*) malloc(M_local*N_local*sizeof(double));
    int v = M_local*N_local*rank;
    for (int j = 0; j < N_local; j++) {
        for (int i = 0; i < M_local; i++) {
            values[i + j*N_local] = (double) v;
            v++;
        }
    }
    MatSetValues(mat_local, M_local, row_indices, N_local, col_indices, values, INSERT_VALUES);
    MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY);

    // Create local matrices on global communicator
    Mat mat_global;
    IS is_row;
    int idx[4] = {0, 1, 2, 3};
    ISCreateGeneral(MPI_COMM_WORLD, M_local, idx, PETSC_COPY_VALUES, &is_row);
    MatCreateSubMatrix(mat_local, is_row, NULL, MAT_INITIAL_MATRIX, &mat_global);

    // View each local mat on global communicator (sleep for `rank` seconds so output is ordered)
    sleep(rank);
    MatView(mat_global, 0);

    // Create merged mat on global communicator
    //    For this test, I just put the four locally computed matrices on the diagonal of the merged matrix
    //    In the 4-to-1 merge, this would compute T_merged from T_alpha, T_beta, T_gamma, and T_omega (children)
    int M_merged = M_local*size;
    int N_merged = N_local*size;
    Mat mat_merged;
    MatCreate(MPI_COMM_WORLD, &mat_merged);
    MatSetSizes(mat_merged, PETSC_DECIDE, PETSC_DECIDE, M_merged, N_merged);
    MatSetFromOptions(mat_merged);

    // Get values of local matrix to put on diagonal
    double* values_diag = (double*) malloc(M_local*N_local*sizeof(double));
    MatGetValues(mat_global, M_local, row_indices, N_local, col_indices, values_diag);

    // Put local matrix contributions into merged matrix (placeholder for computing merged matrix)
    for (int i = 0; i < M_local; i++) {
        row_indices[i] = i + M_local*rank;
    }

    for (int j = 0; j < N_local; j++) {
        col_indices[j] = j + N_local*rank;
    }
    MatSetValues(mat_merged, M_local, row_indices, N_local, col_indices, values_diag, INSERT_VALUES);
    MatAssemblyBegin(mat_merged, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat_merged, MAT_FINAL_ASSEMBLY);

    // View merged mat on global communicator
    sleep(rank);
    MatView(mat_merged, 0);

    // Clean up
    free(row_indices);
    free(col_indices);
    free(values);
    free(values_diag);
    MatDestroy(&mat_local);
    MatDestroy(&mat_global);
    MatDestroy(&mat_merged);

}
#endif

void TEST_petsc_matrix_axpy() {

    int M = 8;
    int N = 10;
    double alpha = 2.0;
    ParallelMatrix<double> X(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N);
    ParallelMatrix<double> Y(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N);

    Vector<int> X_row_indices = {0, 2, 4, 6};
    Vector<int> X_col_indices = vectorRange(0, N-1);
    Matrix<double> X_values(X_row_indices.size(), X_col_indices.size());
    int v = 0;
    for (int i = 0; i < X_row_indices.size(); i++) {
        int ii = X_row_indices[i];
        for (int j = 0; j < X_col_indices.size(); j++) {
            int jj = X_col_indices[j];
            X_values(i, j) = (double) v;
            v++;
        }
    }
    X.setValues(X_row_indices, X_col_indices, X_values, INSERT_VALUES);
    X.beginAssembly(MAT_FINAL_ASSEMBLY);

    Vector<int> Y_row_indices = {1, 3, 5, 7};
    Vector<int> Y_col_indices = vectorRange(0, N-1);
    Matrix<double> Y_values(Y_row_indices.size(), Y_col_indices.size());
    int w = 0;
    for (int i = 0; i < Y_row_indices.size(); i++) {
        int ii = Y_row_indices[i];
        for (int j = 0; j < Y_col_indices.size(); j++) {
            int jj = Y_col_indices[j];
            Y_values(i, j) = (double) w;
            w++;
        }
    }
    Y.setValues(Y_row_indices, Y_col_indices, Y_values, INSERT_VALUES);
    Y.beginAssembly(MAT_FINAL_ASSEMBLY);

    X.endAssembly(MAT_FINAL_ASSEMBLY);
    Y.endAssembly(MAT_FINAL_ASSEMBLY);

    MatAXPY(Y.mat, alpha, X.mat, DIFFERENT_NONZERO_PATTERN);

    MatView(Y.mat, PETSC_VIEWER_STDOUT_WORLD);

}

int main(void) {

    MPI_Init(nullptr, nullptr);
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    // TEST_petsc_matrix_serial_to_parallel();
    TEST_petsc_matrix_comm();
    // TEST_petsc_matrix_init();

    // MPI_Barrier(MPI_COMM_WORLD);
    // TEST_petsc_matrix_axpy();

    PetscFinalize();
    MPI_Finalize();

}