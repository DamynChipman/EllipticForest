#ifndef QUAD_NODE_HPP_
#define QUAD_NODE_HPP_

#include <cstddef>
#include "EllipticForestApp.hpp"
#include "MPI.hpp"

namespace EllipticForest {

template<typename T>
class Node : public MPI::MPIObject {

public:

	/**
	 * @brief Data stored in this node
	 * 
	 */
	T data;

	/**
	 * @brief Path of this node in the quadtree
	 * 
	 */
	std::string path;

	/**
	 * @brief Level of refinement
	 * 
	 */
	int level;

	/**
	 * @brief First rank that owns this node
	 * 
	 */
	int pfirst;

	/**
	 * @brief Last rank that owns this node
	 * 
	 */
	int plast;

	/**
	 * @brief Flag for if this node is a leaf
	 * 
	 */
    bool leaf = false;

	/**
	 * @brief Communicator that includes only the ranks this node lives of (pfrist - plast)
	 * 
	 */
	MPI::Communicator node_comm;

	/**
	 * @brief Construct a new Node object (default)
	 * 
	 */
	Node() :
		MPIObject(MPI_COMM_WORLD)
			{}

	/**
	 * @brief Construct a new Node object on a communicator
	 * 
	 * @param comm 
	 */
	Node(MPI_Comm comm) :
		MPIObject(comm)
			{}

	/**
	 * @brief Construct a new Node object from the data and metadata
	 * 
	 * @param comm MPI communicator
	 * @param data Data to store in node
	 * @param path Path of the node in quadtree
	 * @param level Level of refinement
	 * @param pfirst First rank this node lives on
	 * @param plast Last rank this node lives on
	 */
    Node(MPI_Comm comm, T data, std::string path, int level, int pfirst, int plast) :
        MPIObject(comm),
        data(data),
        path(path),
        level(level),
        pfirst(pfirst),
        plast(plast) {

		// Create node communicator that is a subset of the tree communicator
		// MPI::communicatorSubsetRange(comm, pfirst, plast, 20+level, &node_comm);

	}

	/**
	 * @brief Checks if this node is owned by this rank
	 * 
	 * @return true 
	 * @return false 
	 */
	bool isOwned() {
		return pfirst <= this->getRank() && this->getRank() <= plast;
	}

	// /**
	//  * @brief 
	//  * 
	//  * @param newGroup 
	//  * @param newComm 
	//  */
	// void getMPIGroupComm(MPI_Group* newGroup, MPI_Comm* newComm) {
	// 	// auto& app = EllipticForest::EllipticForestApp::getInstance();
	// 	MPI_Group group; MPI_Comm_group(this->getComm(), &group);
	// 	int ranges[1][3] = {pfirst, plast, 1};
	// 	MPI_Group_range_incl(group, 1, ranges, newGroup);
	// 	MPI_Comm_create_group(this->getComm(), *newGroup, 20+level, newComm);
	// 	// int color = isOwned() ? 0 : MPI_UNDEFINED;
	// 	// app.log("color = %i", color);
	// 	// MPI_Comm_split(this->getComm(), color, this->getRank(), newComm);
	// }

	// void freeMPIGroupComm(MPI_Group* group, MPI_Comm* comm) {
	// 	MPI_Group_free(group);
	// 	MPI_Comm_free(comm);
	// }

	/**
	 * @brief Writes this node to a string
	 * 
	 * @return std::string 
	 */
	std::string str() {
		std::string out = "";
		out += "node: path = " + path + ", level = " + std::to_string(level) + ", ranks = [" + std::to_string(pfirst) + "-" + std::to_string(plast) + "]";
		return out;
	}

	/**
	 * @brief Writes node to an ostream
	 * 
	 * @param os Ostream reference to write to
	 * @param node Node to write
	 * @return std::ostream& 
	 */
	friend std::ostream& operator<<(std::ostream& os, const Node<T>& node) {
		os << "node: path = " << node.path << ", level = " << node.level << ", ranks = [" << node.pfirst << "-" << node.plast << "]" << std::endl;
		return os;
	}

};

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

} // NAMESPACE : EllipticForest

#endif // QUAD_NODE_HPP_