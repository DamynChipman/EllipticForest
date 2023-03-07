#include <cmath>
#include <iostream>
#include <utility>
#include <string>

#include <mpi.h>

#include <EllipticForestApp.hpp>
#include <P4est.hpp>

int refine_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
    if (quadrant->level >= 3) {
        return 0;
    }

    // Get bounds of quadrant
    double vxyz[3];
    double xLower, xUpper, yLower, yUpper;
    p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
    xLower = vxyz[0];
    yLower = vxyz[1];

    p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
    xUpper = vxyz[0];
    yUpper = vxyz[1];

    // Refine if middle of domain
    // Refine if -0.2 < x,y < 0.2
    double xMiddle = (xLower + xUpper) / 2.0;
    double yMiddle = (yLower + yUpper) / 2.0;
    if (xMiddle > -0.6 && xMiddle < 0.6 && yMiddle > -0.6 && yMiddle < 0.6) {
        return 1;
    }
    else {
        return 0;
    }
}

int main(int argc, char** argv) {

    sc_MPI_Init(&argc, &argv);
    EllipticForest::EllipticForestApp app(&argc, &argv);
    int myRank = -1;
    int nRanks = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    app.log("I am rank %i of %i", myRank, nRanks);

    // Create p4est
    int fillUniform = 0;
    int refineRecursive = 1;
    int minLevel = 0;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);

    p4est_refine(p4est, refineRecursive,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
        if (quadrant->level >= 3) {
            return 0;
        }

        // Get bounds of quadrant
        double vxyz[3];
        double xLower, xUpper, yLower, yUpper;
        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
        xLower = vxyz[0];
        yLower = vxyz[1];

        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
        xUpper = vxyz[0];
        yUpper = vxyz[1];

        // Refine if middle of domain
        // Refine if -0.2 < x,y < 0.2
        double xMiddle = (xLower + xUpper) / 2.0;
        double yMiddle = (yLower + yUpper) / 2.0;
        if (xMiddle > -0.6 && xMiddle < 0.6 && yMiddle > -0.6 && yMiddle < 0.6) {
            return 1;
        }
        else {
            return 0;
        }
    },
    NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);

    // Save mesh
    std::string filename = "toybox_mesh";
    p4est_vtk_write_file(p4est, NULL, filename.c_str());

    return EXIT_SUCCESS;
}