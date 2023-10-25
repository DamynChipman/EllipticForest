#include <Vector.hpp>

using namespace EllipticForest;

int rank, size;

void TEST_petsc_vector_init() {
    
    int N = 4*size;
    ParallelVector<double> vec(MPI_COMM_WORLD, PETSC_DECIDE, N);

    Vector<int> indices(4);
    Vector<double> values(4);
    for (int i = 0; i < 4; i++) {
        indices[i] = i + 4*rank;
        values[i] = (double) i + 4*rank;
    }
    
    vec.setValues(indices, values, INSERT_VALUES);

    vec.beginAssembly();
    vec.endAssembly();

    VecView(vec.vec, PETSC_VIEWER_STDOUT_WORLD);

}

void TEST_petsc_vector_serial_to_parallel() {

    int N = 4*size;
    Vector<double> serial_vector(N);
    for (int i = 0; i < N; i++) {
        serial_vector[i] = (double) i;
    }

    ParallelVector<double> parallel_vector(MPI_COMM_WORLD, serial_vector);
    parallel_vector.beginAssembly();
    parallel_vector.endAssembly();

    VecView(parallel_vector.vec, 0);

}

void TEST_petsc_vector_waxpy() {

    int N = 4*size;
    double alpha = 1.0;
    ParallelVector<double> w(MPI_COMM_WORLD, PETSC_DECIDE, N);
    ParallelVector<double> x(MPI_COMM_WORLD, PETSC_DECIDE, N);
    ParallelVector<double> y(MPI_COMM_WORLD, PETSC_DECIDE, N);

    Vector<int> x_indices(4);
    Vector<double> x_values(4);
    for (int i = 0; i < 4; i++) {
        x_indices[i] = i + 4*rank;
        x_values[i] = (double) i + 4*rank;
    }
    x.setValues(x_indices, x_values, INSERT_VALUES);
    x.beginAssembly();

    Vector<int> y_indices(4);
    Vector<double> y_values(4);
    for (int i = 0; i < 4; i++) {
        y_indices[i] = i + 4*rank;
        y_values[i] = (double) i + 4*rank;
    }
    y.setValues(y_indices, y_values, INSERT_VALUES);
    y.beginAssembly();

    x.endAssembly();
    y.endAssembly();

    VecWAXPY(w.vec, alpha, x.vec, y.vec);

    VecView(w.vec, PETSC_VIEWER_STDOUT_WORLD);

}

int main(void) {

    MPI_Init(nullptr, nullptr);
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_vector_init();

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_vector_serial_to_parallel();

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_petsc_vector_waxpy();

    PetscFinalize();
    MPI_Finalize();

}