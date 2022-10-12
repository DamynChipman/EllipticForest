#include "P4est.hpp"

namespace EllipticForest {

namespace p4est {

p4est_t* p4est_new_uniform(MPI_comm mpicomm, std::size_t levels, std::size_t data_size, p4est_init_t init_fn, void* user_pointer) {

    // // Create new p4est connectivity
    // p4est_connectivity_t* p4est_connectivity_new_unitsquare();

    // // Create new p4est
    // p4est_t* p4est_new()

}

} // NAMESPACE : p4est

} // NAMESPACE : EllipticForest