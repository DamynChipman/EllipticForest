#include "MPI.hpp"

namespace EllipticForest {

namespace MPI {

template<>
int send<std::string>(std::string& str, int dest, int tag, MPI_Comm comm) {
    int size = str.length();
    send(size, dest, tag, comm);
    return MPI_Send(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), dest, tag, comm);
}

template<>
int receive<std::string>(std::string& str, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    int size;
    receive(size, src, tag, comm, status);
    str.resize(size);
    return MPI_Recv(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), src, tag, comm, status);
}

template<>
int broadcast<std::string>(std::string& str, int root, MPI_Comm comm) {
    // std::cout << "Calling string broadcast" << std::endl;
    int rank; MPI_Comm_rank(comm, &rank);
    int size = str.length();
    broadcast(size, root, comm);
    if (rank != root) str = std::string(TypeTraits<std::string>::getAddress(str), size);
    return MPI_Bcast(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), root, comm);
}

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest