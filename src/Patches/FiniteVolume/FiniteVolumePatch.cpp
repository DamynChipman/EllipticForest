#include "FiniteVolumePatch.hpp"

namespace EllipticForest {

FiniteVolumePatch::FiniteVolumePatch() :
    MPIObject(MPI_COMM_WORLD)
        {}

FiniteVolumePatch::FiniteVolumePatch(MPI::Communicator comm) : 
    MPIObject(comm),
    grid_(comm, 0, 0, 0, 0, 0, 0)
        {}

FiniteVolumePatch::FiniteVolumePatch(MPI::Communicator comm, FiniteVolumeGrid grid) :
    MPIObject(comm),
    grid_(grid)
        {}

FiniteVolumePatch::~FiniteVolumePatch() {
    // printf("[RANK %i/%i] Calling FiniteVolumePatch destructor.\n", this->getRank(), this->getSize());
}

std::string FiniteVolumePatch::name() {
    return "FiniteVolumePatch";
}

FiniteVolumeGrid& FiniteVolumePatch::grid() {
    return grid_;
}

Matrix<double>& FiniteVolumePatch::matrixX() {
    return X_serial;
}

Matrix<double>& FiniteVolumePatch::matrixH() {
    return H_serial;
}

Matrix<double>& FiniteVolumePatch::matrixS() {
    return S_serial;
}

Matrix<double>& FiniteVolumePatch::matrixT() {
    return T_serial;
}

Vector<double>& FiniteVolumePatch::vectorU() {
    return u_serial;
}

Vector<double>& FiniteVolumePatch::vectorG() {
    return g_serial;
}

Vector<double>& FiniteVolumePatch::vectorV() {
    return v_serial;
}

Vector<double>& FiniteVolumePatch::vectorF() {
    return f_serial;
}

Vector<double>& FiniteVolumePatch::vectorH() {
    return h_serial;
}

Vector<double>& FiniteVolumePatch::vectorW() {
    return w_serial;
}

double FiniteVolumePatch::dataSize() {
    double BYTE_2_MEGABYTE = 1024*1024;
    double size_MB = (4*sizeof(int) + sizeof(bool)) / BYTE_2_MEGABYTE;

    size_MB += (T_serial.nRows() * T_serial.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (H_serial.nRows() * H_serial.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (S_serial.nRows() * S_serial.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (X_serial.nRows() * X_serial.nCols() * sizeof(double)) / BYTE_2_MEGABYTE;

    size_MB += (u_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (g_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (v_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (f_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (h_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;
    size_MB += (w_serial.size() * sizeof(double)) / BYTE_2_MEGABYTE;

    return size_MB;
}

namespace MPI {

template<>
int broadcast(FiniteVolumePatch& patch, int root, MPI::Communicator comm) {
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    // app.log("Broadcasting 0...");
    broadcast(patch.nCoarsens, root, comm);
    // app.log("Broadcasting 1...");
    broadcast(patch.grid(), root, comm);
    // app.log("Broadcasting 2...");
    broadcast(patch.matrixX(), root, comm);
    // app.log("Broadcasting 3...");
    broadcast(patch.matrixH(), root, comm);
    // app.log("Broadcasting 4...");
    broadcast(patch.matrixS(), root, comm);
    // app.log("Broadcasting 5...");
    broadcast(patch.matrixT(), root, comm);
    // app.log("Broadcasting 6...");
    broadcast(patch.vectorU(), root, comm);
    // app.log("Broadcasting 7...");
    broadcast(patch.vectorG(), root, comm);
    // app.log("Broadcasting 8...");
    broadcast(patch.vectorV(), root, comm);
    // app.log("Broadcasting 9...");
    broadcast(patch.vectorF(), root, comm);
    // app.log("Broadcasting 10...");
    broadcast(patch.vectorH(), root, comm);
    // app.log("Broadcasting 11...");
    broadcast(patch.vectorW(), root, comm);
    return 1;
}

} // NAMESPACE : EllipticForest

} // NAMESPACE : EllipticForest