#ifndef QUAD_NODE_HPP_
#define QUAD_NODE_HPP_

#include <cstddef>
#include "EllipticForestApp.hpp"
#include "MPI.hpp"

namespace EllipticForest {

using NodePathKey = std::string;

// Forward declaration of QuadNode for AbstractNodeFactory
template<typename T> class Node;

template<typename T>
class AbstractNodeFactory {
public:
	virtual Node<T>* createNode(T data, std::string path, int level, int pfirst, int plast) = 0;
	virtual Node<T>* createChildNode(Node<T>* parentNode, int siblingID, int pfirst, int plast) = 0;
	virtual Node<T>* createParentNode(std::vector<Node<T>*> childrenNodes, int pfirst, int plast) = 0;
};

template<typename T>
class Node : public MPI::MPIObject {

public:

	T data;
	std::string path;
	int level;
	int pfirst;
	int plast;
    bool leaf = false;
	MPI::Communicator node_comm;

	Node() :
		MPIObject(MPI_COMM_WORLD)
			{}

	Node(MPI_Comm comm) :
		MPIObject(comm)
			{}

    Node(MPI_Comm comm, T data, std::string path, int level, int pfirst, int plast) :
        MPIObject(comm),
        data(data),
        path(path),
        level(level),
        pfirst(pfirst),
        plast(plast) {

		// Create node communicator that is a subset of the tree communicator
		MPI::communicatorSubsetRange(comm, pfirst, plast, 20+level, &node_comm);

	}

	bool isOwned() {
		return pfirst <= this->getRank() && this->getRank() <= plast;
	}

	void getMPIGroupComm(MPI_Group* newGroup, MPI_Comm* newComm) {
		MPI_Group group; MPI_Comm_group(this->getComm(), &group);
		int ranges[1][3] = {pfirst, plast, 1};
		MPI_Group_range_incl(group, 1, ranges, newGroup);
		MPI_Comm_create_group(this->getComm(), *newGroup, 20+level, newComm);
	}

	void freeMPIGroupComm(MPI_Group* group, MPI_Comm* comm) {
		MPI_Group_free(group);
		MPI_Comm_free(comm);
	}

	std::string str() {
		std::string out = "";
		out += "node: path = " + path + ", level = " + std::to_string(level) + ", ranks = [" + std::to_string(pfirst) + "-" + std::to_string(plast) + "]";
		return out;
	}

	friend std::ostream& operator<<(std::ostream& os, const Node<T>& node) {
		os << "node: path = " << node.path << ", level = " << node.level << ", ranks = [" << node.pfirst << "-" << node.plast << "]" << std::endl;
		return os;
	}

};

// template<class T>
// struct MPI::TypeTraits<Node<T>> {

//     static inline MPI_Datatype getType(Node<T>& node) {
//         MPI_Datatype tempType, nodeType;
//         MPI_Aint lb, extent;
//         int count = 5;
//         const int blocklengths[5] = {
//             (int) MPI::TypeTraits<T>::getSize(node.data),
//             (int) MPI::TypeTraits<std::string>::getSize(node.path),
//             (int) MPI::TypeTraits<int>::getSize(node.level),
//             (int) MPI::TypeTraits<int>::getSize(node.pfirst),
//             (int) MPI::TypeTraits<int>::getSize(node.plast)
//         };
//         const MPI_Aint displacements[5] = {
//             offsetof(Node<T>, data),
//             offsetof(Node<T>, path),
//             offsetof(Node<T>, level),
//             offsetof(Node<T>, data),
//             offsetof(Node<T>, data)
//         };
//         const MPI_Datatype types[5] = {
//             MPI::TypeTraits<T>::getType(node.data),
//             MPI::TypeTraits<std::string>::getType(node.path),
//             MPI::TypeTraits<int>::getType(node.level),
//             MPI::TypeTraits<int>::getType(node.pfirst),
//             MPI::TypeTraits<int>::getType(node.plast)
//         };
//         MPI_Type_create_struct(count, blocklengths, displacements, types, &tempType);
//         MPI_Type_get_extent(tempType, &lb, &extent);
//         MPI_Type_create_resized(tempType, lb, extent, &nodeType);
//         MPI_Type_commit(&nodeType);
//         return nodeType;
//     }
//     static inline std::size_t getSize(Node<T>& node) { return 1; }
//     static inline Node<T>* getAddress(Node<T>& node) { return &node; }

// };

namespace MPI {

template<class T>
int send(Node<T>& node, int dest, int tag, MPI_Comm comm) {
    send(node.data, dest, tag, comm);
    send(node.path, dest, tag, comm);
    send(node.level, dest, tag, comm);
    send(node.pfirst, dest, tag, comm);
    send(node.plast, dest, tag, comm);
    send(node.leaf, dest, tag, comm);
    return 0;
}

template<class T>
int receive(Node<T>& node, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    receive(node.data, src, tag, comm, status);
    receive(node.path, src, tag, comm, status);
    receive(node.level, src, tag, comm, status);
    receive(node.pfirst, src, tag, comm, status);
    receive(node.plast, src, tag, comm, status);
    receive(node.leaf, src, tag, comm, status);
    return 0;
}

template<class T>
int broadcast(Node<T>& node, int root, MPI_Comm comm) {
    broadcast(node.data, root, comm);
    broadcast(node.path, root, comm);
    broadcast(node.level, root, comm);
    broadcast(node.pfirst, root, comm);
    broadcast(node.plast, root, comm);
    broadcast(node.leaf, root, comm);
    return 0;
}

}

// template<class T>
// int MPI::broadcast_(Node<T>& node, int dest, int tag, MPI_Comm comm, Node<T>*) {
//     std::cout << "Calling Node broadcast" << std::endl;
//     return 0;
// }

// template<>
// int send<Node<T>>(Node<T>& node, int dest, int tag, MPI_Comm comm) {
//     MPI::send(node.data, dest, tag + 1, comm);
//     MPI::send(node.path, dest, tag + 2, comm);
//     MPI::send(node.level, dest, tag + 3, comm);
//     MPI::send(node.pfirst, dest, tag + 4, comm);
//     MPI::send(node.plast, dest, tag + 5, comm);
//     return 0;
// }

// template<>
// int receive<Node<T>>(Node<T>& node, int src, int tag, MPI_Comm comm, MPI_Status* status) {
//     MPI::receive(node.data, src, tag + 1, comm, status);
//     MPI::receive(node.path, src, tag + 2, comm, status);
//     MPI::receive(node.level, src, tag + 3, comm, status);
//     MPI::receive(node.pfirst, src, tag + 4, comm, status);
//     MPI::receive(node.plast, src, tag + 5, comm, status);
//     return 0;
// }

} // NAMESPACE : EllipticForest

#endif // QUAD_NODE_HPP_