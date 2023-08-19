#ifndef P4EST_HPP_
#define P4EST_HPP_

#include <string>
#include <mpi.h>
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_connectivity.h>
#include <p4est_vtk.h>
#include <p4est_search.h>
#include <p4est_bits.h>
#include <p4est_communication.h>

namespace EllipticForest {

namespace p4est {

/**
 * @brief Create a connectivity for a square domain
 * 
 * @param x_lower Lower x-coordinate
 * @param x_upper Upper x-coordinate
 * @param y_lower Lower y-coordinate
 * @param y_upper Upper y-coordinate
 * @return p4est_connectivity_t* 
 */
p4est_connectivity_t* p4est_connectivity_new_square_domain(double x_lower, double x_upper, double y_lower, double y_upper);

/**
 * @brief Returns the path of the quadrant as a string
 * 
 * @param q Quadrant
 * @return std::string 
 */
std::string p4est_quadrant_path(const p4est_quadrant_t* q);

} // NAMESPACE : p4est

} // NAMESPACE : EllipticForest

#endif // P4EST_HPP_