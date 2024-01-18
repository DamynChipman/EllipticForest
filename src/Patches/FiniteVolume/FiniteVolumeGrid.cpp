#include "FiniteVolumeGrid.hpp"

namespace EllipticForest {

FiniteVolumeGrid::FiniteVolumeGrid() :
    MPIObject(MPI_COMM_SELF)
        {}

FiniteVolumeGrid::FiniteVolumeGrid(MPI::Communicator comm, int nx, double x_lower, double x_upper, int ny, double y_lower, double y_upper) :
    MPIObject(comm),
    nx_(nx),
    x_lower_(x_lower),
    x_upper_(x_upper),
    dx_((x_upper - x_lower)/nx),
    ny_(ny),
    y_lower_(y_lower),
    y_upper_(y_upper),
    dy_((y_upper - y_lower)/ny)
        {}

FiniteVolumeGrid::~FiniteVolumeGrid()
    {}

double FiniteVolumeGrid::point(DimensionIndex dim, int index) {
    double p = 0;
    if (dim == DimensionIndex::X) {
        p = (x_lower_ + dx_/2) + index*dx_;
    }
    else if (dim == DimensionIndex::Y) {
        p = (y_lower_ + dy_/2) + index*dy_;
    }
    else {
        throw std::invalid_argument("Invalid `dim` argument. TODO: Add better error messages");
    }
    return p;
}

std::string FiniteVolumeGrid::name() {
    return "FiniteVolumeGrid";
}

std::size_t FiniteVolumeGrid::nx() {
    return nx_;
}

std::size_t FiniteVolumeGrid::ny() {
    return ny_;
}

double FiniteVolumeGrid::xLower() {
    return x_lower_;
}

double FiniteVolumeGrid::xUpper() {
    return x_upper_;
}

double FiniteVolumeGrid::yLower() {
    return y_lower_;
}

double FiniteVolumeGrid::yUpper() {
    return y_upper_;
}

double FiniteVolumeGrid::dx() {
    return dx_;
}

double FiniteVolumeGrid::dy() {
    return dy_;
}

double FiniteVolumeGrid::operator()(std::size_t DIM, std::size_t index) {
    return point((DimensionIndex) DIM, index);
}

namespace MPI {

template<>
int broadcast(FiniteVolumeGrid& grid, int root, MPI::Communicator comm) {
    int nx = grid.nx();
    int ny = grid.ny();
    double xLower = grid.xLower();
    double xUpper = grid.xUpper();
    double yLower = grid.yLower();
    double yUpper = grid.yUpper();
    broadcast(nx, root, comm);
    broadcast(ny, root, comm);
    broadcast(xLower, root, comm);
    broadcast(xUpper, root, comm);
    broadcast(yLower, root, comm);
    broadcast(yUpper, root, comm);
    int rank; MPI_Comm_rank(comm, &rank);
    grid = FiniteVolumeGrid(comm, nx, xLower, xUpper, ny, yLower, yUpper);
    return 1;
}

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest