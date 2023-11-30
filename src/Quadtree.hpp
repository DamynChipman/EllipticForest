#ifndef QUADTREE_HPP_
#define QUADTREE_HPP_

#include <iostream>
#include <string>
#include <ostream>
#include <vector>
#include <functional>
#include <set>

#include "EllipticForestApp.hpp"
#include "MPI.hpp"
#include "P4est.hpp"
#include "QuadNode.hpp"

namespace EllipticForest {

enum NodeCommunicationPolicy {
	BROADCAST,
	STRIPE
};

template<typename T>
class Quadtree : public MPI::MPIObject {

public:

	using NodeMap = std::map<NodePathKey, Node<T>*>;
	NodeMap map;
	p4est_t* p4est;
	T* rootDataPtr_;
	AbstractNodeFactory<T>* nodeFactory;
	std::function<int(Node<T>*)> visitNodeFn;
	std::function<void(T&)> visitNodeDataFn;
	std::function<int(Node<T>*, std::vector<Node<T>*>)> visitFamilyFn;
	NodeCommunicationPolicy communicationPolicy = NodeCommunicationPolicy::BROADCAST;

	Quadtree() :
		MPIObject(MPI_COMM_WORLD),
		rootDataPtr_(nullptr),
		nodeFactory(nullptr)
			{}
	
	Quadtree(MPI_Comm comm, p4est_t* p4est, T& rootData, AbstractNodeFactory<T>& nodeFactory) :
		MPIObject(comm),
		p4est(p4est),
		rootDataPtr_(&rootData),
		nodeFactory(&nodeFactory) {

		auto& app = EllipticForest::EllipticForestApp::getInstance();
		// app.log("Creating quadtree...");

		// Save user pointer and store self in it
		void* savedUserPointer = p4est->user_pointer;
		p4est->user_pointer = reinterpret_cast<void*>(this);

		// Use p4est_search_all to do depth first traversal and create quadtree map and nodes
		int callPost = 0;
		p4est_search_all(
			p4est,
			callPost,
			[](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {

				// Get quadtree
				auto& app = EllipticForest::EllipticForestApp::getInstance();
				auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
				auto& map = quadtree.map;
				// auto& node_set = quadtree.node_set;

				// Get MPI info
				int rank;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);

				// Compute unique path
				std::string path = p4est::p4est_quadrant_path(quadrant);
				// node_set.insert(path);
				// app.log("path = " + path);

				// Create the node communicator
				MPI::Communicator node_comm;
				MPI::communicatorSubsetRange(MPI_COMM_WORLD, pfirst, plast, 0, &node_comm);

				// Check if quadrant is owned by this rank
				bool owned = pfirst <= rank && rank <= plast;

				// If owned, create Node in map
				if (owned) {
					// app.log("In quadrant: %s", path.c_str());
					// Create node and metadata
					Node<T>* node = new Node<T>;
					// Node<T> node;

					// Create node's data
					if (quadrant->level == 0) {
						// Quadrant is root; init with root data
						node = quadtree.nodeFactory->createNode(*quadtree.rootDataPtr_, path, quadrant->level, pfirst, plast);
					}
					else {
						// Quadrant is not root; init from parent and sibling index
						Node<T>* parentNode = quadtree.getParentFromPath(path);
						int siblingID = p4est_quadrant_child_id(quadrant);
						node = quadtree.nodeFactory->createChildNode(parentNode, siblingID, pfirst, plast);
					}

					// Additional info for node
					node->leaf = local_num >= 0;
					node->node_comm = node_comm;

					// Put in map
					map[path] = node;

					
					// int index = quadtree.nodeIndex(path);
					// app.log("  Inserting node with path: %s", path.c_str());
					// map.insert(index, node);
					// quadtree.insert(path, node);
				}
				else {
					// Not owned by local rank, put in nullptr for this node
					map[path] = nullptr;
				}

				return 1;
			},
			NULL,
			NULL
		);

		// Restore p4est user pointer
		p4est->user_pointer = savedUserPointer;

		// app.log("Done with quadtree constructor.");

	}

	~Quadtree() {
		// Iterate through map and delete any owned nodes
		// for (typename NodeMap::iterator iter = map.begin(); iter != map.end(); ++iter) {
		// 	if (iter->second != nullptr) {
		// 		delete iter->second;
		// 	}
		// }
	}

	Quadtree& operator=(Quadtree&& other) {
		if (this != &other) {
			map = other.map;
			// Set map seconds to nullptr so destructor doesn't delete them
			for (typename NodeMap::iterator iter = other.map.begin(); iter != other.map.end(); ++iter) {
				iter->second = nullptr;
			}
			p4est = other.p4est;
			rootDataPtr_ = other.rootDataPtr_;
			nodeFactory = other.nodeFactory;
		}
		return *this;
	}

	// Quadtree(Quadtree&& other) :
	// 	MPIObject(other->getComm()),
	// 	map(other.map),
	// 	p4est(other.p4est),
	// 	rootDataPtr_(other.rootDataPtr_),
	// 	nodeFactory(other.nodeFactory) {
	// 			std::cout << "MOVE CONSTRUCTOR CALLED" << std::endl;
	// 		}

	// int nodeIndex(NodePathKey key) {
	// 	int index;
	// 	auto b = node_set.begin();
	// 	auto p = node_set.find(key);
	// 	if (p == node_set.end()) {
	// 		index =  -1;
	// 	}
	// 	else {
	// 		index = std::distance(b, p);
	// 	}
	// 	return index;
	// }

	// void insert(NodePathKey key, Node<T>& node) {
	// 	// 
	// 	node_set.insert(key);

	// 	// 
	// 	int index;
	// 	auto b = node_set.begin();
	// 	auto p = node_set.find(key);
	// 	if (p == node_set.end()) {
	// 		index =  -1;
	// 	}
	// 	else {
	// 		index = std::distance(b, p);
	// 	}

	// 	// 
	// 	map.insert(index, node);

	// }

	// Node<T>& obtain(NodePathKey key) {

	// }

	void traversePreOrder(std::function<int(Node<T>*)> visit) {

		p4est->user_pointer = (void*) this;
		visitNodeFn = visit;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			p4est_search_visit,
			NULL,
			NULL,
			NULL
		);

	}

	void traversePreOrder(std::function<void(T&)> visit) {

		visitNodeDataFn = visit;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			p4est_search_visit_data,
			NULL,
			NULL,
			NULL
		);

	}

	void traversePostOrder(std::function<int(Node<T>*)> visit) {

		visitNodeFn = visit;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			NULL,
			p4est_search_visit,
			NULL,
			NULL
		);

	}

	void traversePostOrder(std::function<void(T&)> visit) {

		visitNodeDataFn = visit;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			NULL,
			p4est_search_visit_data,
			NULL,
			NULL
		);

	}

	void merge(std::function<int(Node<T>*)> visitLeaf, std::function<int(Node<T>*, std::vector<Node<T>*>)> visitFamily) {

		visitNodeFn = visitLeaf;
		visitFamilyFn = visitFamily;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			NULL,
			p4est_search_merge,
			NULL,
			NULL
		);

	}

	void split(std::function<int(Node<T>*)> visitLeaf, std::function<int(Node<T>*, std::vector<Node<T>*>)> visitFamily) {

		visitNodeFn = visitLeaf;
		visitFamilyFn = visitFamily;
		int skipLevels = 0;
		p4est_search_reorder(
			p4est,
			skipLevels,
			NULL,
			p4est_search_merge,
			NULL,
			NULL,
			NULL
		);

	}

	Node<T>* getParentFromPath(std::string path) {
		std::string parent_path = path.substr(0, path.length() - 1);
		// auto index = std::distance(node_set.begin(), node_set.find(parent_path));
		// auto i = map.find(index);
		// return map.value_at(i);
		return map[parent_path];
	}

	T& root() {
		return map["0"]->data;
	}

	static int p4est_search_visit(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto path = p4est::p4est_quadrant_path(quadrant);
		auto* node = map[path];
		int cont = 1;
		if (node != nullptr) {
			bool owned = node->pfirst <= rank && rank <= node->plast;
			if (owned) {
				cont = quadtree.visitNodeFn(node);
			}
		}
		return cont;
	}

	static int p4est_search_visit_data(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		if (node != nullptr) {
			bool owned = node->pfirst <= rank && rank <= node->plast;
			if (owned) {
				quadtree.visitNodeDataFn(node->data);
			}
		}
		return cont;
	}

	static int p4est_search_merge(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
		// std::cout << "[p4est_search_merge] user_pointer = " << p4est->user_pointer << std::endl;
		// auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
		// auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		// auto& quadtree = *quadtree_pointer<T>;
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		bool owned = node->pfirst <= rank && rank <= node->plast;
		// std::string ppath;
		// if (node == nullptr) {
		// 	ppath = "NULL";
		// }
		// else {
		// 	ppath = node->path;
		// }
		// app.log("Post-quadrant callback - " + node->str());
		// app.log("pfirst = %i, plast = %i, owned = %i", node->pfirst, node->plast, owned);

		// printf("[RANK %i / %i]\n", rank, ranks);
		// std::cout << "\tlocal_num = " << local_num << std::endl;
		// std::cout << "\tnode = " << node << std::endl;
		
		// Get node group and communicator
		// MPI_Group nodeGroup;
		MPI_Comm nodeComm;
		nodeComm = node->node_comm;
		// node->getMPIGroupComm(&nodeGroup, &nodeComm);
		int nodeRank; MPI_Comm_rank(nodeComm, &nodeRank);

		// app.log(node->str() + " - " + "comm = " + MPI::communicatorGetName(nodeComm));

		if (local_num >= 0) {
			// Quadrant is a leaf; call leaf callback
			if (node != nullptr && owned) {
				cont = quadtree.visitNodeFn(node);
				return cont;
			}
		}
		else {
			// Quadrant is a branch; call branch callback
			std::vector<Node<T>*> children{4};
			if (node->pfirst == node->plast) {
				// Node is owned by single rank
				for (int i = 0; i < 4; i++) {
					children[i] = map[node->path + std::to_string(i)];
				}
			}
			else if (owned) {
				// Nodes on different ranks; broadcast to ranks that own the node
				if (quadtree.communicationPolicy == NodeCommunicationPolicy::BROADCAST) {

					

					// int comm_name_size = 0;
					// char comm_name_buffer[256];
					// MPI_Comm_get_name(nodeComm, comm_name_buffer, &comm_name_size);
					// std::string comm_name(comm_name_buffer, comm_name_size);
					// app.log("comm name = " + comm_name);

					// Iterate through children
					for (int i = 0; i < 4; i++) {
						Node<T>* child = map[node->path + std::to_string(i)];
						
						// Communicate root rank to all others in node comm
						bool amRoot = child != nullptr;
						int pnode;
						int maybeRoot = (amRoot ? nodeRank : 0);
						// app.log("CALL MPI_Allreduce");
						MPI_Allreduce(&maybeRoot, &pnode, 1, MPI_INT, MPI_MAX, nodeComm);

						// Allocate memory on my rank to store incoming node data
						// app.log("HERE 1");
						if (child == nullptr) {
							// child = quadtree.nodeFactory->createChildNode(node, i, node->pfirst, node->plast);
							child = new Node<T>(node->getComm());
						}
						// app.log("HERE 2");

						// Broadcast node
						
						// app.log("Broadcasting child node %i, root = %i", i, pnode);
						MPI::broadcast(*child, pnode, nodeComm);
						
						// Store child in children
						children[i] = child;
					}

					// node->freeMPIGroupComm(&nodeGroup, &nodeComm);
				}
				else if (quadtree.communicationPolicy == NodeCommunicationPolicy::STRIPE) {
					// TODO
				}
			}

			// If owned, call family branch callback
			if (owned) {
				cont = quadtree.visitFamilyFn(node, children);
			}

		}
		return cont;
	}

	static int p4est_search_split(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {

		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
		auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		bool owned = node->pfirst <= rank && rank <= node->plast;

		if (local_num >= 0) {
			// Quadrant is a leaf; call leaf callback
			if (node != nullptr && owned) {
				cont = quadtree.visitNodeFn(node);
				return cont;
			}
		}
		else {
			// Quadrant is a branch; call branch callback
			std::vector<Node<T>*> children{4};
			if (node->pfirst == node->plast) {
				// Node is owned by single rank
				for (int i = 0; i < 4; i++) {
					children[i] = map[node->path + std::to_string(i)];
				}
			}
			else if (owned) {
				// Nodes on different ranks; broadcast to ranks that own the node
				// Get node group and communicator
				MPI_Group nodeGroup;
				MPI_Comm nodeComm;
				node->getMPIGroupComm(&nodeGroup, &nodeComm);
				int nodeRank; MPI_Comm_rank(nodeComm, &nodeRank);

				// Iterate through children
				for (int i = 0; i < 4; i++) {
					Node<T>* child = map[node->path + std::to_string(i)];
					
					// Communicate root rank to all others in node comm
					bool amRoot = child != nullptr;
					int pnode;
					int maybeRoot = (amRoot ? nodeRank : 0);
					MPI_Allreduce(&maybeRoot, &pnode, 1, MPI_INT, MPI_MAX, nodeComm);

					// Allocate memory on my rank to store incoming node data
					if (child == nullptr) {
						child = new Node<T>(node->getComm());
					}

					// Broadcast node
					MPI::broadcast(*child, pnode, nodeComm);
					
					// Store child in children
					children[i] = child;
				}
			}

			// If owned, call family branch callback
			if (owned) {
				cont = quadtree.visitFamilyFn(node, children);
			}
		}
		return cont;

	}

};

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_