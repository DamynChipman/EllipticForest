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
    dy_((y_upper - y_lower)/ny) {

    //
    create();
    setFromOptions();
    setup();

}

FiniteVolumeGrid::~FiniteVolumeGrid() {
    if (is_created_) {
        DMDestroy(&dm);
    }
}

Petsc::ErrorCode FiniteVolumeGrid::create() {
    is_created_ = true;
    int dof_vertex = 0;
    int dof_face = 0;
    int dof_element = 1;
    int stencil_width = 1;
    return DMStagCreate2d(this->getComm(), DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx_, ny_, PETSC_DECIDE, PETSC_DECIDE, dof_vertex, dof_face, dof_element, DMSTAG_STENCIL_STAR, stencil_width, nullptr, nullptr, &dm);
}

Petsc::ErrorCode FiniteVolumeGrid::setFromOptions() {
    return DMSetFromOptions(dm);
}

Petsc::ErrorCode FiniteVolumeGrid::setup() {
    return DMSetUp(dm);
}

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

} // NAMESPACE : EllipticForest