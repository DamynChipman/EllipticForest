#include <vector>
#include <MPI.hpp>
#include <Vector.hpp>

using namespace EllipticForest;

int rank, size;

void TEST_std_vector() {
    int data_size = 4;
    std::vector<int> local = {0 + rank*data_size, 1 + rank*data_size, 2 + rank*data_size, 3 + rank*data_size};
    std::vector<int> gathered(data_size*size);

    printf("[RANK %i/%i] local    = [%i, %i, %i, %i, ]\n", rank, size, local[0], local[1], local[2], local[3]);

    MPI::allgather(local, gathered, data_size, MPI_COMM_WORLD);

    printf("[RANK %i/%i] gathered = [", rank, size);
    for (auto& i : gathered)
        printf("%i, ", i);
    printf("]\n");
}

void TEST_ef_vector() {
    int data_size = 4;
    Vector<int> local = {0 + rank*data_size, 1 + rank*data_size, 2 + rank*data_size, 3 + rank*data_size};
    Vector<int> gathered(data_size*size);

    printf("[RANK %i/%i] local    = [%i, %i, %i, %i, ]\n", rank, size, local[0], local[1], local[2], local[3]);

    MPI::allgather(local, gathered, data_size, MPI_COMM_WORLD);

    printf("[RANK %i/%i] gathered = [", rank, size);
    for (auto& i : gathered.data())
        printf("%i, ", i);
    printf("]\n");
}

int main(void) {

    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MPI::HEAD_RANK) {
        printf("===== TEST_std_vector =====\n");
    }
    TEST_std_vector();
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MPI::HEAD_RANK) {
        printf("===== TEST_ef_vector =====\n");
    }
    TEST_ef_vector();
    MPI_Barrier(MPI_COMM_WORLD);

    return MPI_Finalize();
}