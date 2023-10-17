#include <vector>
#include <MPI.hpp>

using namespace EllipticForest;

int rank, size;

void TEST_std_vector() {
    std::vector<int> local(rank);
    for (auto& i : local) {
        i = rank;
    }
    printf("[RANK %i/%i] local = [", rank, size);
    for (auto& i : local) printf("%i, ", i);
    printf("]\n");

    std::vector<int> recv_count_local = {rank};
    std::vector<int> recv_counts(size);
    MPI::allgather(recv_count_local, recv_counts, 1, MPI_COMM_WORLD);
    printf("[RANK %i/%i] recv_counts = [", rank, size);
    for (auto& i : recv_counts) printf("%i, ", i);
    printf("]\n");

    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; i++) {
        int d = 0;
        for (int j = 0; j < i; j++) {
            d += recv_counts[j];
        }
        displacements[i] = d;
    }
    printf("[RANK %i/%i] displacements = [", rank, size);
    for (auto& i : displacements) printf("%i, ", i);
    printf("]\n");

    int gathered_size = 0;
    for (int i = 0; i < size; i++) {
        gathered_size += i;
    }
    std::vector<int> gathered(gathered_size);
    MPI::allgatherv(local, gathered, recv_counts, displacements, MPI_COMM_WORLD);
    printf("[RANK %i/%i] gathered = [", rank, size);
    for (auto& i : gathered) printf("%i, ", i);
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

    return MPI_Finalize();
}