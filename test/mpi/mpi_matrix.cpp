#include <Matrix.hpp>

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
    TEST_petsc_matrix_init();

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_matrix_axpy();

    PetscFinalize();
    MPI_Finalize();

}