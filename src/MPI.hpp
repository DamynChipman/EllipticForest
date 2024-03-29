#ifndef MPI_HPP_
#define MPI_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

namespace EllipticForest {

namespace MPI {

/**
 * @brief Global head rank of zero
 * 
 */
static int HEAD_RANK = 0;

/**
 * @brief Type alias for MPI_Comm
 * 
 */
using Communicator = MPI_Comm;

/**
 * @brief Type alias for MPI_Group
 * 
 */
using Group = MPI_Group;

/**
 * @brief Type alias for MPI_Status
 * 
 */
using Status = MPI_Status;

/**
 * @brief Type alias for MPI_Datatype
 * 
 */
using Datatype = MPI_Datatype;

/**
 * @brief Class for any MPI based object; can also be used by itself for global MPI utility
 * 
 */
class MPIObject {

protected:

    /**
     * @brief MPI communicator of object
     * 
     */
    Communicator comm_;

    /**
     * @brief MPI rank
     * 
     */
    int rank_;

    /**
     * @brief MPI size
     * 
     */
    int size_;

public:

    /**
     * @brief Construct a new MPIObject object with default MPI_COMM_WORLD communicator
     * 
     */
    MPIObject() :
        comm_(MPI_COMM_WORLD) {

        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);

    }

    /**
     * @brief Construct a new MPIObject object with specified communicator
     * 
     * @param comm MPI communicator
     */
    MPIObject(Communicator comm) :
        comm_(comm) {

        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);

    }

    /**
     * @brief Returns the MPI communicator
     * 
     * @return const Communicator 
     */
    virtual const Communicator getComm() const { return comm_; }

    /**
     * @brief Returns the MPI rank
     * 
     * @return const int 
     */
    virtual const int getRank() const { return rank_; }

    /**
     * @brief Returns the MPI size of the communicator
     * 
     * @return const int 
     */
    virtual const int getSize() const { return size_; }

};

/**
 * @brief Creates a sub communicator from a range of ranks on the super communicator
 * 
 * @param super_comm Existing super communicator
 * @param r_first First rank on super communicator
 * @param r_last Last rank on super communicator
 * @param tag Message tag
 * @param sub_comm [out] Reference to declared, but not created, sub communicator
 */
void communicatorSubsetRange(Communicator super_comm, int r_first, int r_last, int tag, Communicator* sub_comm);

/**
 * @brief Sets the name of the communicator
 * 
 * @param comm MPI communicator
 * @param name Name to set
 */
void communicatorSetName(Communicator comm, const std::string& name);

/**
 * @brief Gets the name of the communicator
 * 
 * @param comm MPI communicator
 * @return std::string 
 */
std::string communicatorGetName(Communicator comm);

/**
 * @brief MPI type traits for primitive, C++, and user defined objects
 * 
 * This struct provides an interface for templated MPI calls to get the corresponding data types,
 * as well as getting the necessary data (data type, size, and address) for the communication.
 * 
 * To expand to user defined types, create template specializations for each of the functions that
 * need to differ from the default.
 * 
 * @tparam T The type of the object
 */
template<class T>
struct TypeTraits {

    /**
     * @brief Returns the MPI datatype of the object
     * 
     * @param data Reference to data
     * @return Datatype 
     */
    static inline Datatype getType(T& data);

    /**
     * @brief Returns the size of the data; defaults to 1
     * 
     * @param data Reference to data
     * @return std::size_t 
     */
    static inline std::size_t getSize(T& data) { return 1; }

    /**
     * @brief Returns the address of the start of the data
     * 
     * @param data Reference to data
     * @return T* 
     */
    static inline T* getAddress(T& data) { return &data; }

};

// Specialization of the MPITypeTraits for primitive types 
#define PRIMITIVE(Type, MpiType) \
	template<> \
	inline Datatype TypeTraits<Type>::getType(Type&) { \
		return MpiType; \
	}
PRIMITIVE(char, 				MPI_CHAR);
PRIMITIVE(wchar_t,				MPI_WCHAR);
PRIMITIVE(short, 				MPI_SHORT);
PRIMITIVE(int, 					MPI_INT);
PRIMITIVE(long, 				MPI_LONG);
PRIMITIVE(signed char, 			MPI_SIGNED_CHAR);
PRIMITIVE(unsigned char, 		MPI_UNSIGNED_CHAR);
PRIMITIVE(unsigned short, 		MPI_UNSIGNED_SHORT);
PRIMITIVE(unsigned int,			MPI_UNSIGNED);
PRIMITIVE(unsigned long,		MPI_UNSIGNED_LONG);
PRIMITIVE(unsigned long long,	MPI_UNSIGNED_LONG_LONG);
PRIMITIVE(float, 				MPI_FLOAT);
PRIMITIVE(double, 				MPI_DOUBLE);
PRIMITIVE(long double,			MPI_LONG_DOUBLE);
PRIMITIVE(bool,                 MPI_CXX_BOOL);
#undef PRIMATIVE

/**
 * @brief Wrapper for MPI_Send
 * 
 * @tparam T Type of object to communicate
 * @param data Reference to data
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @return int
 */
template<class T>
int send(T& data, int dest, int tag, Communicator comm) {
    return MPI_Send(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), dest, tag, comm);
}

/**
 * @brief Wrapper for MPI_Recv
 * 
 * @tparam T Type of object to communicate
 * @param data Reference to data
 * @param src Source rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @param status MPI status
 * @return int 
 */
template<class T>
int receive(T& data, int src, int tag, Communicator comm, Status* status) {
    return MPI_Recv(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), src, tag, comm, status);
}

/**
 * @brief Wrapper for MPI_Bcast
 * 
 * @tparam T Type of object to communicate
 * @param data Reference to data
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int broadcast(T& data, int root, Communicator comm) {
    return MPI_Bcast(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), root, comm);
}

/**
 * @brief Wrapper for MPI_Allreduce
 * 
 * @tparam T Type of object to communicate
 * @param send Reference of data to send
 * @param recv Reference of data to receive
 * @param op MPI operator
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allreduce(T& send, T& recv, MPI_Op op, Communicator comm) {
    assert(TypeTraits<T>::getSize(send) == TypeTraits<T>::getSize(recv));
    return MPI_Allreduce(TypeTraits<T>::getAddress(send), TypeTraits<T>::getAddress(recv), TypeTraits<T>::getSize(send), TypeTraits<T>::getType(send), op, comm);
}

/**
 * @brief Wrapper for MPI_Allgather
 * 
 * @tparam T Type of object to communicate
 * @param send_data Reference of data to send
 * @param recv_data Storage for data to receive from involved ranks
 * @param recv_count Count of elements received from any rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgather(T& send_data, T& recv_data, int recv_count, Communicator comm) {
    return MPI_Allgather(TypeTraits<T>::getAddress(send_data), TypeTraits<T>::getSize(send_data), TypeTraits<T>::getType(send_data), TypeTraits<T>::getAddress(recv_data), recv_count, TypeTraits<T>::getType(recv_data), comm);
}

/**
 * @brief Wrapper for MPI_Allgatherv
 * 
 * @tparam T Type of object to communicate
 * @param send_data Reference of data to send
 * @param recv_data Storage for data to receive from involved ranks
 * @param recv_counts Vector of counts of elements received from ranks
 * @param displacements Vector of displacements of elements received from ranks to put into `recv_data`
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgatherv(T& send_data, T& recv_data, std::vector<int> recv_counts, std::vector<int> displacements, Communicator comm) {
    return MPI_Allgatherv(TypeTraits<T>::getAddress(send_data), TypeTraits<T>::getSize(send_data), TypeTraits<T>::getType(send_data), TypeTraits<T>::getAddress(recv_data), recv_counts.data(), TypeTraits<T>::getType(recv_data), displacements.data(), comm);
}

/**
 * @brief Specialization of std::vector<T> for TypeTraits
 * 
 * @tparam T Type of vector elements
 */
template<class T>
struct TypeTraits<std::vector<T>> {

    /**
     * @brief Returns the MPI datatype: type of T
     * 
     * @param vec Reference to std::vector
     * @return Datatype 
     */
    static inline Datatype getType(std::vector<T>& vec) { return TypeTraits<T>::getType(vec[0]); }

    /**
     * @brief Returns the size of the std::vector
     * 
     * @param vec Reference to std::vector
     * @return std::size_t 
     */
    static inline std::size_t getSize(std::vector<T>& vec) { return vec.size(); }

    /**
     * @brief Returns the beginning address of the data in `vec`
     * 
     * @param vec Reference to std::vector
     * @return T* 
     */
    static inline T* getAddress(std::vector<T>& vec) { return &vec.front(); }

};

/**
 * @brief Function overload of @sa `send` for std::vector<T>
 * 
 * @tparam T Type of data to communicate
 * @param data Reference to data
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int send(std::vector<T>& data, int dest, int tag, Communicator comm) {
    int size = data.size();
    send(size, dest, tag, comm);
    return MPI_Send(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), dest, tag, comm);
}

/**
 * @brief Function overload of @sa `receive` for std::vector<T>
 * 
 * @tparam T Type of data to communicate
 * @param data Reference to data
 * @param src Source rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @param status MPI status
 * @return int 
 */
template<class T>
int receive(std::vector<T>& data, int src, int tag, Communicator comm, Status* status) {
    int size;
    receive(size, src, tag, comm, status);
    data.resize(size);
    return MPI_Recv(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), src, tag, comm, status);
}

/**
 * @brief Function overload of @sa `broadcast` for std::vector<T>
 * 
 * @tparam T Type of data to communicate
 * @param data Reference to data
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int broadcast(std::vector<T>& data, int root, Communicator comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    int size = data.size();
    broadcast(size, root, comm);
    if (rank != root) data.resize(size);
    return MPI_Bcast(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), root, comm);
}

/**
 * @brief Function overload of @sa `allgather` for std::vector<T>
 * 
 * @tparam T Type of object in vector
 * @param send_data Reference to vector to send
 * @param recv_data Storage for vector to receive from involved ranks
 * @param recv_count Count of elements received from any rank
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgather(std::vector<T>& send_data, std::vector<T>& recv_data, int recv_count, Communicator comm) {
    return MPI_Allgather(TypeTraits<std::vector<T>>::getAddress(send_data), TypeTraits<std::vector<T>>::getSize(send_data), TypeTraits<std::vector<T>>::getType(send_data), TypeTraits<std::vector<T>>::getAddress(recv_data), recv_count, TypeTraits<std::vector<T>>::getType(recv_data), comm);
}

/**
 * @brief Function overload of @sa `allgatherv` for std::vector<T>
 * 
 * @tparam T Type of object in vector
 * @param send_data Reference of vector to send
 * @param recv_data Storage for vector to receive from involved ranks
 * @param recv_counts Vector of counts of elements received from ranks
 * @param displacements Vector of displacements of elements received from ranks to put into `recv_data`
 * @param comm MPI communicator
 * @return int 
 */
template<class T>
int allgatherv(std::vector<T>& send_data, std::vector<T>& recv_data, std::vector<int> recv_counts, std::vector<int> displacements, Communicator comm) {
    return MPI_Allgatherv(TypeTraits<std::vector<T>>::getAddress(send_data), TypeTraits<std::vector<T>>::getSize(send_data), TypeTraits<std::vector<T>>::getType(send_data), TypeTraits<std::vector<T>>::getAddress(recv_data), recv_counts.data(), displacements.data(), TypeTraits<std::vector<T>>::getType(recv_data), comm);
}

/**
 * @brief Specialization of std::string for TypeTraits
 * 
 */
template<>
struct TypeTraits<std::string> {

    /**
     * @brief Returns the MPI datatype: MPI_CHAR
     * 
     * @param str Reference to std::string
     * @return Datatype 
     */
    static inline Datatype getType(std::string& str) { return MPI_CHAR; }

    /**
     * @brief Returns the size of the std::string
     * 
     * @param str Reference to std::string
     * @return std::size_t 
     */
    static inline std::size_t getSize(std::string& str) { return str.length(); }

    /**
     * @brief Returns the address of the std::string
     * 
     * @param str Reference to std::string
     * @return char* 
     */
    static inline char* getAddress(std::string& str) { return str.data(); }

};

/**
 * @brief Template overload for @sa `send` for std::string
 * 
 * @param str Reference to std::string
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @return int 
 */
template<>
int send<std::string>(std::string& str, int dest, int tag, Communicator comm);

/**
 * @brief Template overload for @sa `receive` for std::string
 * 
 * @param str Reference to std::string
 * @param src Source rank
 * @param tag Message tag
 * @param comm MPI communicator
 * @param status MPI status
 * @return int 
 */
template<>
int receive<std::string>(std::string& str, int src, int tag, Communicator comm, Status* status);

/**
 * @brief Template overload for @sa `broadcast` for std::string
 * 
 * @param str Reference to std::string
 * @param root Root rank
 * @param comm MPI communicator
 * @return int 
 */
template<>
int broadcast<std::string>(std::string& str, int root, Communicator comm);

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest

#endif // MPI_HPP_