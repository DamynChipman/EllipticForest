#ifndef FINITE_VOLUME_PATCH_HPP_
#define FINITE_VOLUME_PATCH_HPP_

#include "FiniteVolumeGrid.hpp"
#include "FiniteVolumeSolver.hpp"
#include "../../Patch.hpp"

namespace EllipticForest {

class FiniteVolumePatch : public MPI::MPIObject, public PatchBase<FiniteVolumePatch, FiniteVolumeGrid, FiniteVolumeSolver, double> {

protected:

    Matrix<double> X_serial{}, H_serial{}, S_serial{}, T_serial{};
    Vector<double> u_serial{}, g_serial{}, v_serial{}, f_serial{}, h_serial{}, w_serial{};
    ParallelMatrix<double> X_parallel{}, H_parallel{}, S_parallel{}, T_parallel{};
    ParallelVector<double> u_parallel{}, g_parallel{}, v_parallel{}, f_parallel{}, h_parallel{}, w_parallel{};
    FiniteVolumeGrid grid_;

public:

    FiniteVolumePatch();
    FiniteVolumePatch(MPI::Communicator comm);
    FiniteVolumePatch(MPI::Communicator comm, FiniteVolumeGrid grid);

    virtual std::string name();
    virtual FiniteVolumeGrid& grid();
    virtual Matrix<double>& matrixX();
    virtual Matrix<double>& matrixH();
    virtual Matrix<double>& matrixS();
    virtual Matrix<double>& matrixT();
    virtual Vector<double>& vectorU();
    virtual Vector<double>& vectorG();
    virtual Vector<double>& vectorV();
    virtual Vector<double>& vectorF();
    virtual Vector<double>& vectorH();
    virtual Vector<double>& vectorW();

    double dataSize();
    std::string str();    

};

namespace MPI {

template<>
int broadcast(FiniteVolumePatch& patch, int root, MPI::Communicator comm);

} // NAMESPACE : EllipticForest

} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_PATCH_HPP_