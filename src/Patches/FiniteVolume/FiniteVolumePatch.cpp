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

std::string FiniteVolumePatch::str() {
    std::string res;

    res += "nCoarsens = " + std::to_string(nCoarsens) + "\n";

    res += "grid:\n";
    res += "  nx = " + std::to_string(grid().nx()) + "\n";
    res += "  ny = " + std::to_string(grid().ny()) + "\n";
    res += "  xLower = " + std::to_string(grid().xLower()) + "\n";
    res += "  xUpper = " + std::to_string(grid().xUpper()) + "\n";
    res += "  yLower = " + std::to_string(grid().yLower()) + "\n";
    res += "  yUpper = " + std::to_string(grid().yUpper()) + "\n";

    res += "data:\n";
    res += "  X = [" + std::to_string(matrixX().nRows()) + ", " + std::to_string(matrixX().nCols()) + "]\n";
    res += "  S = [" + std::to_string(matrixS().nRows()) + ", " + std::to_string(matrixS().nCols()) + "]\n";
    res += "  T = [" + std::to_string(matrixT().nRows()) + ", " + std::to_string(matrixT().nCols()) + "]\n";
    res += "  u = [" + std::to_string(vectorU().size()) + "]\n";
    res += "  g = [" + std::to_string(vectorG().size()) + "]\n";
    res += "  v = [" + std::to_string(vectorV().size()) + "]\n";
    res += "  f = [" + std::to_string(vectorF().size()) + "]\n";
    res += "  h = [" + std::to_string(vectorH().size()) + "]\n";
    res += "  w = [" + std::to_string(vectorW().size()) + "]\n";

    return res;
}

namespace MPI {

template<>
int broadcast(FiniteVolumePatch& patch, int root, MPI::Communicator comm) {
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    broadcast(patch.nCoarsens, root, comm);
    broadcast(patch.grid(), root, comm);
    
    // app.log("HERE 1");
    // if (patch.par_matrix_T.mat == NULL) {
    //     app.log("HERE 2");
    //     patch.par_matrix_T = ParallelMatrix<double>(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, 0, 0, MATMPIDENSE);
    // }
    // else {
    //     app.log("HERE 3");
    // }
    // app.log("HERE 4");
    // patch.par_matrix_T = ParallelMatrix<double>(comm, patch.par_matrix_T);
    
    // Only broadcast meta data, matrices and vectors are local (maybe...)
    broadcast(patch.matrixX(), root, comm);
    broadcast(patch.matrixH(), root, comm);
    broadcast(patch.matrixS(), root, comm);
    broadcast(patch.matrixT(), root, comm);
    broadcast(patch.vectorU(), root, comm);
    broadcast(patch.vectorG(), root, comm);
    broadcast(patch.vectorV(), root, comm);
    broadcast(patch.vectorF(), root, comm);
    broadcast(patch.vectorH(), root, comm);
    broadcast(patch.vectorW(), root, comm);
    return 1;
}

} // NAMESPACE : EllipticForest

} // NAMESPACE : EllipticForest