#ifndef MPI_HPP_
#define MPI_HPP_

#include <mpi.h>

namespace EllipticForest {

class MPIObject {

protected:

    MPI_Comm comm_;
    int rank_;
    int size_;

public:

    MPIObject(MPI_Comm comm) :
        comm_(comm) {

        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);

    }

    virtual const MPI_Comm getComm() const { return comm_; }
    virtual const int getRank() const { return rank_; }
    virtual const int getSize() const { return size_; }

};

} // NAMESPACE : EllipticForest

#endif // MPI_HPP_