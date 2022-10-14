#ifndef P4EST_HPP_
#define P4EST_HPP_

#include <mpi.h>
#include <p4est.h>
#include <p4est_connectivity.h>

namespace EllipticForest {

namespace p4est {

p4est_t* p4est_new_uniform(MPI_Comm mpicomm, std::size_t levels, std::size_t data_size, p4est_init_t init_fn, void* user_pointer);

} // NAMESPACE : p4est

} // NAMESPACE : EllipticForest

#endif // P4EST_HPP_