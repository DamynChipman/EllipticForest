#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    EllipticForest::MPI::MPIObject mpiobj;

    EllipticForest::Matrix<double> mat;
    if (mpiobj.getRank() == 0) {
        mat = EllipticForest::Matrix<double>(3, 2,
            {0.0, 1.0,
             2.0, 4.0,
             8.0, 16.0}
        );
        // EllipticForest::MPI::send(mat, 1, 0, mpiobj.getComm());
    }
    // else {
    //     EllipticForest::MPI::receive(mat, 0, 0, mpiobj.getComm(), nullptr);
    // }

    std::this_thread::sleep_for(std::chrono::seconds(mpiobj.getRank()));
    // std::cout << mpiobj << mat << std::endl;
    EllipticForest::MPI::broadcast(mat, 0, mpiobj.getComm());
    std::cout << mpiobj << mat << std::endl;

    MPI_Finalize();

    return EXIT_SUCCESS;
}