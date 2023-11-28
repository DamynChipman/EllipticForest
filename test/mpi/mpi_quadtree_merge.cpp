#include <Quadtree.hpp>
#include <Matrix.hpp>
#include <petscmat.h>

using namespace EllipticForest;

int rank, size;

struct TestNode {
    ParallelMatrix<double> A;
    ParallelVector<double> b;
};

void TEST_parallel_matrix_quadtree() {

    // Create the node map
    std::map<std::string, TestNode*> node_map;

    // Create the root of the tree; everyone does this
    std::string root_path = "0";
    node_map[root_path] = new TestNode;

    // Create children nodes; each rank does it's own
    std::string child_path = "0";
    child_path += std::to_string(rank);

    // Put child node in tree; each rank has it's own
    node_map[child_path] = new TestNode;

    int nrows = 16;
    int ncols = 16;
    node_map[child_path]->A = ParallelMatrix<double>(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols);

    printf("[RANK %i/%i] node_map[%s].A = %p\n", rank, size, child_path.c_str(), node_map[child_path]->A.mat);
    if (node_map[child_path]->A.mat == NULL) {
        printf("[RANK %i/%i] node_map[%s].A is NULL\n", rank, size, child_path.c_str());
    }

    // Clean up
    delete node_map[root_path];
    delete node_map[child_path];

}

int main(int argc, char** argv) {

    MPI_Init(nullptr, nullptr);
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TEST_parallel_matrix_quadtree();

    PetscFinalize();
    MPI_Finalize();

}