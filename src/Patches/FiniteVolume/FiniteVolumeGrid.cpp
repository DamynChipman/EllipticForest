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
    // create();
    // setFromOptions();
    // setup();

}

// FiniteVolumeGrid::FiniteVolumeGrid(FiniteVolumeGrid& copy_grid) :
//     MPIObject(copy_grid.getComm()),
//     nx_(copy_grid.nx()),
//     x_lower_(copy_grid.xLower()),
//     x_upper_(copy_grid.xUpper()),
//     dx_((copy_grid.xUpper() - copy_grid.xLower())/copy_grid.nx()),
//     ny_(copy_grid.ny()),
//     y_lower_(copy_grid.yLower()),
//     y_upper_(copy_grid.yUpper()),
//     dy_((copy_grid.yUpper() - copy_grid.yLower())/copy_grid.ny()),
//     dm(copy_grid.dm)
//         {
//             printf("[RANK %i/%i] Calling FiniteVolumeGrid copy constructor.\n", this->getRank(), this->getSize());
//         }
    
// FiniteVolumeGrid::FiniteVolumeGrid(FiniteVolumeGrid&& move_grid) :
//     MPIObject(move_grid.getComm()),
//     nx_(move_grid.nx()),
//     x_lower_(move_grid.xLower()),
//     x_upper_(move_grid.xUpper()),
//     dx_((move_grid.xUpper() - move_grid.xLower())/move_grid.nx()),
//     ny_(move_grid.ny()),
//     y_lower_(move_grid.yLower()),
//     y_upper_(move_grid.yUpper()),
//     dy_((move_grid.yUpper() - move_grid.yLower())/move_grid.ny()),
//     dm(move_grid.dm)
//         {
//             printf("[RANK %i/%i] Calling FiniteVolumeGrid move constructor.\n", this->getRank(), this->getSize());
//         }

FiniteVolumeGrid::~FiniteVolumeGrid() {
    // printf("[RANK %i/%i] Calling FiniteVolumeGrid destructor.\n", this->getRank(), this->getSize());
    // if (dm == PETSC_NULLPTR) {
    //     // printf("[RANK %i/%i] Destroying DM...\n", this->getRank(), this->getSize());
    //     DMDestroy(&dm);
    // }
}

// FiniteVolumeGrid& FiniteVolumeGrid::operator=(FiniteVolumeGrid& other) {
//     printf("[RANK %i/%i] Calling FiniteVolumeGrid copy assignment operator.\n", this->getRank(), this->getSize());
//     this->comm_ = other.getComm();
//     this->rank_ = other.getRank();
//     this->size_ = other.getSize();
//     this->nx_ = other.nx();
//     this->x_lower_ = other.xLower();
//     this->x_upper_ = other.xUpper();
//     this->dx_ = other.dx();
//     this->ny_ = other.ny();
//     this->y_lower_ = other.yLower();
//     this->y_upper_ = other.yUpper();
//     this->dy_ = other.dy();
//     this->dm = other.dm;
// }

// FiniteVolumeGrid& FiniteVolumeGrid::operator=(FiniteVolumeGrid&& other) {
//     printf("[RANK %i/%i] Calling FiniteVolumeGrid move assignment operator.\n", this->getRank(), this->getSize());
//     this->comm_ = other.getComm();
//     this->rank_ = other.getRank();
//     this->size_ = other.getSize();
//     this->nx_ = other.nx();
//     this->x_lower_ = other.xLower();
//     this->x_upper_ = other.xUpper();
//     this->dx_ = other.dx();
//     this->ny_ = other.ny();
//     this->y_lower_ = other.yLower();
//     this->y_upper_ = other.yUpper();
//     this->dy_ = other.dy();
//     this->dm = other.dm;
// }

Petsc::ErrorCode FiniteVolumeGrid::create() {
    if (dm == PETSC_NULLPTR) {
        int dof_vertex = 0;
        int dof_face = 0;
        int dof_element = 1;
        int stencil_width = 1;
        return DMStagCreate2d(this->getComm(), DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx_, ny_, PETSC_DECIDE, PETSC_DECIDE, dof_vertex, dof_face, dof_element, DMSTAG_STENCIL_STAR, stencil_width, nullptr, nullptr, &dm);
    }
    else {
        return 0;
    }
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

namespace MPI {

template<>
int broadcast(FiniteVolumeGrid& grid, int root, MPI::Communicator comm) {
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    int nx = grid.nx();
    int ny = grid.ny();
    double xLower = grid.xLower();
    double xUpper = grid.xUpper();
    double yLower = grid.yLower();
    double yUpper = grid.yUpper();
    // app.log("Broadcasting nx...");
    broadcast(nx, root, comm);
    // app.log("Broadcasting ny...");
    broadcast(ny, root, comm);
    // app.log("Broadcasting xlower...");
    broadcast(xLower, root, comm);
    // app.log("Broadcasting xupper...");
    broadcast(xUpper, root, comm);
    // app.log("Broadcasting ylower...");
    broadcast(yLower, root, comm);
    // app.log("Broadcasting yupper...");
    broadcast(yUpper, root, comm);
    // app.log("Creating grid...");
    int rank; MPI_Comm_rank(comm, &rank);
    grid = FiniteVolumeGrid(comm, nx, xLower, xUpper, ny, yLower, yUpper);
    return 1;
}

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest