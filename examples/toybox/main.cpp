#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <EllipticForest.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>

int main(int argc, char** argv) {

    // Create app
    // EllipticForest::EllipticForestApp app(&argc, &argv);
    // app.logHead("Starting toybox...");

    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    EllipticForest::MPIObject mpi(MPI_COMM_WORLD);
    printf("I am rank [%i / %i]\n", mpi.getRank(), mpi.getSize());

    // Create p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    int minLevel = 0;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);

    // Refine the p4est according to the RHS up to the max level
    p4est_refine(p4est, 1,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 2) {
            return 0;
        }

        return p4est_quadrant_touches_corner(quadrant, 0, 1);

    },
    NULL);
    
    p4est_refine(p4est, 1,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 2) {
            return 0;
        }

        return p4est_quadrant_touches_corner(quadrant, 3, 1);

    },
    NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     return 1;
    // },
    // NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     int id = p4est_quadrant_child_id(quadrant);
    //     if (id == 0 || id == 3) {
    //         return 1;
    //     }
    //     else {
    //         return 0;
    //     }
    // },
    // NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     int id = p4est_quadrant_child_id(quadrant);
    //     p4est_quadrant_t* parent;
    //     p4est_quadrant_parent(quadrant, parent);
    //     if (id == 0 && )
    // },
    // NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);
    p4est_partition(p4est, 0, NULL);

    // Save initial mesh
    bool vtkFlag = true;
    if (vtkFlag) {
        std::string VTKFilename = "toybox_mesh";
        p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());
    }

    EllipticForest::Quadtree<double> quadtree{};
    quadtree.buildFromP4est(p4est, 10.0, [&](double& parentData, int childIndex){
        return parentData / 4.0;
    });

    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));
    printf("[RANK %i / %i] Quadtree:\n", mpi.getRank(), mpi.getSize());
    std::cout << quadtree << std::endl;


    MPI_Finalize();

    return EXIT_SUCCESS;
}