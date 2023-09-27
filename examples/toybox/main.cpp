#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <petsc.h>
#include <petscvec.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <EllipticForestApp.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>
#include <P4est.hpp>

template<typename T>
using VectorViewType = Kokkos::View<T*>;

template<typename T>
using MatrixViewType = Kokkos::View<T**>;

class Patch : public EllipticForest::MPI::MPIObject {

public:

    int N = 0;
    Mat A;
    Vec x;

    Patch() :
        MPIObject(MPI_COMM_WORLD)
            {}

    Patch(EllipticForest::MPI::Communicator comm) :
        MPIObject(comm)
            {}

    void create(int N) {
        this->N = N;

        PetscCallVoid(MatCreate(this->getComm(), &A));
        PetscCallVoid(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
        PetscCallVoid(MatSetFromOptions(A));

        PetscCallVoid(VecCreate(this->getComm(), &x));
        PetscCallVoid(VecSetSizes(x, PETSC_DECIDE, N));
        PetscCallVoid(VecSetFromOptions(x));
    }

    void destroy() {
        PetscCallVoid(MatDestroy(&A));
        PetscCallVoid(VecDestroy(&x));
    }

    std::string str() {
        int x_size, A_rows, A_cols, x_size_local, A_rows_local, A_cols_local;
        VecGetSize(x, &x_size);
        MatGetSize(A, &A_rows, &A_cols);
        VecGetLocalSize(x, &x_size_local);
        MatGetLocalSize(A, &A_rows_local, &A_cols_local);
        std::string s = "N = " + std::to_string(N) + ", x = [" + std::to_string(x_size) + " (" + std::to_string(x_size_local) + ")], A = [" + std::to_string(A_rows) + " (" + std::to_string(A_rows_local) + ") x " + std::to_string(A_cols) + " (" + std::to_string(A_cols_local) + ")]";
        return s;
    }

    ~Patch() {}

};

class PatchNodeFactory : public EllipticForest::AbstractNodeFactory<Patch> {

public:

    PatchNodeFactory() {}
    ~PatchNodeFactory() {}

    EllipticForest::Node<Patch> createNode(Patch data, std::string path, int level, int pfirst, int plast) {
        return EllipticForest::Node<Patch>(data.getComm(), data, path, level, pfirst, plast);
    }

    EllipticForest::Node<Patch> createChildNode(EllipticForest::Node<Patch> parent_node, int sibling_id, int pfirst, int plast) {
        auto& parent_patch = parent_node.data;
        Patch child_patch;
        if (parent_node.pfirst == pfirst && parent_node.plast == plast) {
            // Use the same communicator
            child_patch = Patch(parent_node.data.getComm());
        }
        else {
            // Create new communicator
            auto parent_comm = parent_patch.getComm();
            EllipticForest::MPI::Group parent_group;
            EllipticForest::MPI::Group child_group;
            EllipticForest::MPI::Communicator child_comm;
            MPI_Comm_group(parent_comm, &parent_group);
            int ranges[1][3] = {pfirst, plast, 1};
            MPI_Group_range_incl(parent_group, 1, ranges, &child_group);
            MPI_Comm_create_group(parent_comm, child_group, pfirst+plast+sibling_id, &child_comm);
            child_patch = Patch(child_comm);
        }
        child_patch.create(parent_patch.N / 2);
        int idx_lo, idx_hi, N_local;
        VecGetOwnershipRange(child_patch.x, &idx_lo, &idx_hi);
        N_local = idx_hi - idx_lo;
        for (int i = idx_lo; i < idx_hi; i++) {
            VecSetValue(child_patch.x, i, parent_patch.getRank(), INSERT_VALUES);
        }
        VecAssemblyBegin(child_patch.x);
        VecAssemblyEnd(child_patch.x);
        std::string child_path = parent_node.path + std::to_string(sibling_id);
        int child_level = parent_node.level + 1;
        return EllipticForest::Node<Patch>(MPI_COMM_WORLD, child_patch, child_path, child_level, pfirst, plast);
    }

    EllipticForest::Node<Patch> createParentNode(std::vector<EllipticForest::Node<Patch>> children_nodes, int pfirst, int plast) {
        Patch parent_patch;
        parent_patch.create(children_nodes[0].data.N);

        std::string parent_path = children_nodes[0].path.substr(0, children_nodes[0].path.length()-1);
        int parent_level = children_nodes[0].level - 1;
        return EllipticForest::Node<Patch>(MPI_COMM_WORLD, parent_patch, parent_path, parent_level, pfirst, plast);
    }

};

int main(int argc, char** argv) {

    Kokkos::initialize(argc, argv);
    EllipticForest::EllipticForestApp app{&argc, &argv};
    EllipticForest::MPI::MPIObject mpi{};
    app.logHead("Hello from toybox!");

    {
        // Create p4est
        double x_lower = 0;
        double x_upper = 1;
        double y_lower = 0;
        double y_upper = 1;
        p4est_connectivity_t* p4est_conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(x_lower, x_upper, y_lower, y_upper);
        p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, p4est_conn, 0, 2, true, 0, NULL, NULL);
        p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);
        p4est_partition(p4est, 0, NULL);

        // Create root data
        int N = 320;
        Patch root_patch;
        root_patch.create(N);

        // Set values in root data
        int idx_lo, idx_hi, N_local;
        PetscCall(VecGetOwnershipRange(root_patch.x, &idx_lo, &idx_hi));
        N_local = idx_hi - idx_lo;
        for (int i = idx_lo; i < idx_hi; i++) {
            PetscCall(VecSetValue(root_patch.x, i, mpi.getRank(), INSERT_VALUES));
        }
        PetscCall(VecAssemblyBegin(root_patch.x));
        PetscCall(VecAssemblyEnd(root_patch.x));

        // Create node factory
        PatchNodeFactory node_factory;

        // Create quadtree
        EllipticForest::Quadtree<Patch> quadtree(MPI_COMM_WORLD, p4est, root_patch, node_factory);
        auto& map = quadtree.node_map;

        Kokkos::parallel_for(map.capacity(), KOKKOS_LAMBDA(uint32_t i){
            if( map.valid_at(i) ) {
                auto& app = EllipticForest::EllipticForestApp::getInstance();
                auto key   = map.key_at(i);
                auto node = map.value_at(i);
                auto& patch = node.data;
                app.log(node.str() + ", data = " + patch.str());

                // PetscCallVoid(VecView(patch.x, PETSC_VIEWER_STDOUT_SELF));

                patch.destroy();
            }
        });

    }

#if 0
    {
        int N = 16;
        int M = 4;
        Kokkos::UnorderedMap<uint32_t, Patch> map(20);
        Kokkos::parallel_for(M, KOKKOS_LAMBDA(uint32_t i){
            Patch patch;
            patch.create(N);
            for (int i = 0; i < N; i++) {
                PetscCallVoid(VecSetValue(patch.x, i, i, INSERT_VALUES));
            }
            PetscCallVoid(VecAssemblyBegin(patch.x));
            PetscCallVoid(VecAssemblyEnd(patch.x));
            map.insert(i, patch);
        });

        Kokkos::parallel_for(map.capacity(), KOKKOS_LAMBDA(uint32_t i){
            auto& app = EllipticForest::EllipticForestApp::getInstance();
            if( map.valid_at(i) ) {
                auto key   = map.key_at(i);
                auto patch = map.value_at(i);

                PetscCallVoid(VecView(patch.x, PETSC_VIEWER_STDOUT_WORLD));

                patch.destroy();
            }
        });
    }
#endif

#if 0
    auto N = 1600;

    Vec x;
    int block_size = 1;
    auto local_size = PETSC_DECIDE;
    auto global_size = N;
    PetscCall(VecCreate(mpi.getComm(), &x));
    PetscCall(VecSetSizes(x, local_size, global_size));
    PetscCall(VecSetFromOptions(x));

    Vec b;
    PetscCall(VecDuplicate(x, &b));

    Mat A;
    auto local_rows = PETSC_DECIDE;
    auto local_cols = PETSC_DECIDE;
    auto global_rows = N;
    auto global_cols = N;
    PetscCall(MatCreate(mpi.getComm(), &A));
    PetscCall(MatSetSizes(A, local_rows, local_cols, global_rows, global_cols));
    PetscCall(MatSetFromOptions(A));

    int idx_lo, idx_hi, N_local;
    PetscCall(VecGetOwnershipRange(x, &idx_lo, &idx_hi));
    N_local = idx_hi - idx_lo;
    app.log("idx_lo = %i, idx_hi = %i, N_local = %i", idx_lo, idx_hi, N_local);
    int* idx = (int*) malloc(N_local*sizeof(int));
    double* x_values = (double*) malloc(N_local*sizeof(double));
    int local_count = 0;
    for (int i = idx_lo; i < idx_hi; i++) {
        idx[local_count] = i;
        x_values[local_count] = (double) i;
        local_count++;
    }
    PetscCall(VecSetValues(x, N, idx, x_values, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));

    int idx_rows_lo, idx_rows_hi, idx_cols_lo, idx_cols_hi, N_rows_local, N_cols_local;
    PetscCall(MatGetOwnershipRange(A, &idx_rows_lo, &idx_rows_hi));
    N_rows_local = idx_rows_hi - idx_rows_lo;
    // PetscCall(MatGetOwnershipRangeColumn(A, &idx_cols_lo, &idx_cols_hi));
    idx_cols_lo = 0;
    idx_cols_hi = N;
    N_cols_local = N;
    app.log("idx_rows_lo = %i, idx_rows_hi = %i, N_rows_local = %i", idx_rows_lo, idx_rows_hi, N_rows_local);
    app.log("idx_cols_lo = %i, idx_cols_hi = %i, N_cols_local = %i", idx_cols_lo, idx_cols_hi, N_cols_local);
    int* idx_rows = (int*) malloc(N_rows_local*sizeof(int));
    int* idx_cols = (int*) malloc(N_cols_local*sizeof(int));
    double* A_values = (double*) malloc(N_rows_local*N_cols_local*sizeof(double));
    int local_row_count = 0;
    int local_col_count = 0;
    for (int i = idx_rows_lo; i < idx_rows_hi; i++) {
        for (int j = idx_cols_lo; j < idx_cols_hi; j++) {
            idx_rows[local_row_count] = i;
            idx_cols[local_col_count] = j;
            int idx = local_col_count + local_row_count*N_cols_local;
            A_values[idx] = (double) idx;
            local_col_count++;
        }
        local_row_count++;
    }
    PetscCall(MatSetValues(A, global_rows, idx_rows, global_cols, idx_cols, A_values, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    
    PetscCall(VecAssemblyEnd(x));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatMult(A, x, b));

    if (N <= 16) {
        PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(VecView(b, PETSC_VIEWER_STDOUT_WORLD));
    }

    free(idx);
    free(x_values);
    free(idx_rows);
    free(idx_cols);
    free(A_values);
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
#endif

    Kokkos::finalize();
    return EXIT_SUCCESS;
}