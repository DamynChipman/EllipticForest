#ifndef MPI_HPP_
#define MPI_HPP_

#include <mpi.h>

namespace EllipticForest {

namespace MPI {

// Forward declarations
// template<class T> int broadcast(T& data, int root, MPI_Comm comm);
// template<class T> int broadcast_(T& data, int root, MPI_Comm comm, T*);
// template<class T> int broadcast_(std::vector<T>& data, int root, MPI_Comm comm, std::vector<T>*);
// template<> int broadcast_<std::string>(std::string& data, int root, MPI_Comm comm, std::string*);
// template<class T> int send_(T& data, int dest, int tag, MPI_Comm comm, T*);
// template<class T> int receive_(T& data, int src, int tag, MPI_Comm comm, MPI_Status* status, T*);
// template<class T> int broadcast_(T& data, int root, MPI_Comm comm, T*);

class MPIObject {

public:

    MPI_Comm comm;
    int rank;
    int size;

    MPIObject(MPI_Comm comm) :
        comm(comm) {

        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);

    }

    virtual const MPI_Comm getComm() const { return comm; }
    virtual const int getRank() const { return rank; }
    virtual const int getSize() const { return size; }

};

/**
 * @brief MPI type traits for primative, C++, and user defined objects
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

    static inline MPI_Datatype getType(T& raw);
    static inline std::size_t getSize(T& raw) { return 1; }
    static inline T* getAddress(T& raw) { return &raw; }

};

// --=== Template Specializations for TypeTraits ===--
// Specialization of the MPITypeTraits for primitive types 
#define PRIMITIVE(Type, MpiType) \
	template<> \
	inline MPI_Datatype TypeTraits<Type>::getType(Type&) { \
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
#undef PRIMATIVE

template<class T>
int send(T& data, int dest, int tag, MPI_Comm comm) {
    return MPI_Send(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), dest, tag, comm);
}

// template<class T>
// int send_(T& data, int dest, int tag, MPI_Comm comm, T*) {
//     return MPI_Send(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), dest, tag, comm);
// }

template<class T>
int receive(T& data, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    return MPI_Recv(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), src, tag, comm, status);
}

// template<class T>
// int receive_(T& data, int src, int tag, MPI_Comm comm, MPI_Status* status, T*) {
//     return MPI_Recv(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), src, tag, comm, status);
// }

template<class T>
int broadcast(T& data, int root, MPI_Comm comm) {
    // return broadcast_(data, root, comm, static_cast<T*>(nullptr));
    // std::cout << "Calling default broadcast" << std::endl;
    return MPI_Bcast(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), root, comm);
}

template<class T>
int allreduce(T& send, T& recv, MPI_Op op, MPI_Comm comm) {
    assert(TypeTraits<T>::getSize(send) == TypeTraits<T>::getSize(recv));
    return MPI_Allreduce(TypeTraits<T>::getAddress(send), TypeTraits<T>::getAddress(recv), TypeTraits<T>::getSize(send), TypeTraits<T>::getType(send), op, comm);
}

// template<class T>
// int broadcast_(T& data, int root, MPI_Comm comm, T*) {
//     std::cout << "Calling default broadcast" << std::endl;
//     return MPI_Bcast(TypeTraits<T>::getAddress(data), TypeTraits<T>::getSize(data), TypeTraits<T>::getType(data), root, comm);
// }

/**
 * @brief Specialization of std::vector<T> for TypeTraits
 * 
 * @tparam T Type of vector elements
 */
template<class T>
struct TypeTraits<std::vector<T>> {

    static inline MPI_Datatype getType(std::vector<T>& vec) { return TypeTraits<T>::getType(vec[0]); }
    static inline std::size_t getSize(std::vector<T>& vec) { return vec.size(); }
    static inline T* getAddress(std::vector<T>& vec) { return &vec.front(); }

};

// template<class T>
// int send_(std::vector<T>& vec, int d)

// template<class T>
// int send<std::vector<T>>(std::vector<T>& vec, int dest, int tag, MPI_Comm comm) {
//     return MPI_Send(TypeTraits<std::vector<T>>::getAddress(vec), TypeTraits<std::vector<T>>::getSize(vec), TypeTraits<std::vector<T>>::getType(vec), dest, tag, comm);
// }

template<class T>
int send(std::vector<T>& data, int dest, int tag, MPI_Comm comm) {
    int size = data.size();
    send(size, dest, tag, comm);
    return MPI_Send(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), dest, tag, comm);
}

// template<class T>
// int receive<std::vector<T>>(std::vector<T>& data, int src, int tag, MPI_Comm comm, MPI_Status* status) {
//     return MPI_Recv(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), src, tag, comm, status);
// }

template<class T>
int receive(std::vector<T>& data, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    int size;
    receive(size, src, tag, comm, status);
    data.resize(size);
    return MPI_Recv(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), src, tag, comm, status);
}

template<class T>
int broadcast(std::vector<T>& data, int root, MPI_Comm comm) {
    // std::cout << "Calling vector broadcast" << std::endl;
    int rank; MPI_Comm_rank(comm, &rank);
    int size = data.size();
    broadcast(size, root, comm);
    // if (rank != root) data = std::vector<T>(TypeTraits<std::vector<T>>::getAddress(data), size);
    if (rank != root) data.resize(size);
    MPI_Bcast(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), root, comm);
}

// template<typename T>
// int broadcast_(std::vector<T>& data, int root, MPI_Comm comm, std::vector<T>*) {
//     std::cout << "Calling vector broadcast" << std::endl;
//     int rank; MPI_Comm_rank(comm, &rank);
//     int size = data.size();
//     broadcast(size, root, comm);
//     // if (rank != root) data = std::vector<T>(TypeTraits<std::vector<T>>::getAddress(data), size);
//     if (rank != root) data.resize(size);
//     MPI_Bcast(TypeTraits<std::vector<T>>::getAddress(data), TypeTraits<std::vector<T>>::getSize(data), TypeTraits<std::vector<T>>::getType(data), root, comm);
// }

template<>
struct TypeTraits<std::string> {

    static inline MPI_Datatype getType(std::string& str) { return MPI_CHAR; }
    static inline std::size_t getSize(std::string& str) { return str.length(); }
    static inline char* getAddress(std::string& str) { return str.data(); }

};

// template<>
// int receive<std::string>(std::string& str, int src, int tag, MPI_Comm comm, MPI_Status* status) {
//     int count;
//     MPI_Probe(src, tag, comm, status);
//     MPI_Get_count(status, TypeTraits<std::string>::getType(str), &count);
//     str = std::string(TypeTraits<std::string>::getAddress(str), count);
//     return MPI_Recv(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), src, tag, comm, status);
// }

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

// template<>
// int broadcast<std::string>(std::string& str, int root, MPI_Comm comm) {
//     int ret = MPI_Bcast(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), root, comm);
//     str = std::string(TypeTraits<std::string>::getAddress(str));
// }

// class MPISerializable {

// public:

//     MPI_Datatype mpiDatatype;

//     virtual void serialize() = 0;
    
//     void commit() {
//         MPI_Type_commit(&mpiDatatype);
//     }

//     void free() {
//         MPI_Type_free(&mpiDatatype);
//     }

// };

} // NAMESPACE : EllipticForest

} // NAMESPACE : MPI

#endif // MPI_HPP_