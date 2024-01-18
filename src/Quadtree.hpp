/**
 * @file Quadtree.hpp
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Path-indexed quadtree data structure
 */
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
#include "AbstractNodeFactory.hpp"

namespace EllipticForest {

/**
 * @brief Alias for a node path key for the map
 * 
 */
using NodePathKey = std::string;

/**
 * @brief Parallel communication policy for the quadtree nodes
 * BROADCAST: Each node broadcasts its data to other siblings
 * STRIPE: Each node stripes its data to other siblings
 * 
 */
enum NodeCommunicationPolicy {
	BROADCAST,
	STRIPE
};

template<typename T>
class Quadtree : public MPI::MPIObject {

public:

	/**
	 * @brief Alias for a node map
	 * 
	 */
	using NodeMap = std::map<NodePathKey, Node<T>*>;

	/**
	 * @brief Map with storage for local nodes
	 * 
	 */
	NodeMap map;

	/**
	 * @brief Pointer to p4est data structure
	 * 
	 */
	p4est_t* p4est;

	/**
	 * @brief Pointer to root prototype
	 * 
	 */
	T* root_data_ptr_;

	/**
	 * @brief Pointer to node factory for use in p4est routines
	 * 
	 */
	AbstractNodeFactory<T>* node_factory;

	/**
	 * @brief Individual node callback function for use in p4est routines
	 * 
	 */
	std::function<int(Node<T>*)> visit_node_fn;

	/**
	 * @brief Node data callback function for use in p4est routines
	 * 
	 */
	std::function<void(T&)> visit_node_data_fn;

	/**
	 * @brief Node family callback function for use in p4est routines
	 * 
	 */
	std::function<int(Node<T>*, std::vector<Node<T>*>)> visit_family_fn;

	/**
	 * @brief Communication algorithm used based on how merging and splitting are done
	 * 
	 */
	NodeCommunicationPolicy communication_policy = NodeCommunicationPolicy::BROADCAST;

	/**
	 * @brief Default constructor
	 * 
	 */
	Quadtree() :
		MPIObject(MPI_COMM_WORLD),
		root_data_ptr_(nullptr),
		node_factory(nullptr)
			{}
	
	/**
	 * @brief Construct a new Quadtree object from a p4est, prototype root data, and a node factory
	 * 
	 * @param comm MPI communicator
	 * @param p4est Pre-built p4est data structure
	 * @param root_data Prototype for all node construction
	 * @param node_factory Factory class to build new nodes
	 */
	Quadtree(MPI_Comm comm, p4est_t* p4est, T& root_data, AbstractNodeFactory<T>& node_factory) :
		MPIObject(comm),
		p4est(p4est),
		root_data_ptr_(&root_data),
		node_factory(&node_factory) {

		// Save user pointer and store self in it
		void* saved_user_pointer = p4est->user_pointer;
		p4est->user_pointer = reinterpret_cast<void*>(this);

		// Use p4est_search_all to do depth first traversal and create quadtree map and nodes
		int callPost = 0;
		p4est_search_all(
			p4est,
			callPost,
			[](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {

				// Get quadtree
				auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
				auto& map = quadtree.map;

				// Get MPI info
				int rank;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);

				// Compute unique path
				std::string path = p4est::p4est_quadrant_path(quadrant);

				// Create the node communicator
				MPI::Communicator node_comm;
				MPI::communicatorSubsetRange(MPI_COMM_WORLD, pfirst, plast, 0, &node_comm);

				// Check if quadrant is owned by this rank
				bool owned = pfirst <= rank && rank <= plast;

				// If owned, create Node in map
				if (owned) {

					// Create node and metadata
					// Node<T>* node = new Node<T>;
					Node<T>* node;

					// Create node's data
					if (quadrant->level == 0) {
						// Quadrant is root; init with root data
						node = quadtree.node_factory->createNode(*quadtree.root_data_ptr_, path, quadrant->level, pfirst, plast);
					}
					else {
						// Quadrant is not root; init from parent and sibling index
						std::string parent_path = path.substr(0, path.length() - 1);
						Node<T>* parent_node = map[parent_path];
						int siblingID = p4est_quadrant_child_id(quadrant);
						node = quadtree.node_factory->createChildNode(parent_node, siblingID, pfirst, plast);
					}

					// Additional info for node
					node->leaf = local_num >= 0;
					node->node_comm = node_comm;
					node->node_comm_set = true;

					// Put in map
					map[path] = node;

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
		p4est->user_pointer = saved_user_pointer;

	}

	/**
	 * @brief Destroy the Quadtree object; deletes allocated map data
	 * 
	 */
	~Quadtree() {
		// Iterate through map and delete any owned nodes
		// for (typename NodeMap::iterator iter = map.begin(); iter != map.end(); ++iter) {
		// 	if (iter->second != nullptr) {
		// 		delete iter->second;
		// 	}
		// }
	}

	/**
	 * @brief Move assignment operator
	 * 
	 * @param other Quadtree to move
	 * @return Quadtree& 
	 */
	Quadtree& operator=(Quadtree&& other) {
		if (this != &other) {
			map = other.map;
			// Set map seconds to nullptr so destructor doesn't delete them
			for (typename NodeMap::iterator iter = other.map.begin(); iter != other.map.end(); ++iter) {
				iter->second = nullptr;
			}
			p4est = other.p4est;
			root_data_ptr_ = other.root_data_ptr_;
			node_factory = other.node_factory;
		}
		return *this;
	}

	/**
	 * @brief Traverse the quadtree in a pre-order fashion with a node callback function
	 * 
	 * @param visit Node callback function
	 */
	void traversePreOrder(std::function<int(Node<T>*)> visit) {

		p4est->user_pointer = (void*) this;
		visit_node_fn = visit;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			p4est_search_visit,
			NULL,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Traverse the quadtree in a pre-order fashion with a data callback function
	 * 
	 * @param visit Data callback function
	 */
	void traversePreOrder(std::function<void(T&)> visit) {

		visit_node_data_fn = visit;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			p4est_search_visit_data,
			NULL,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Traverse the quadtree in a post-order fashion with a node callback function
	 * 
	 * @param visit Node callback function
	 */
	void traversePostOrder(std::function<int(Node<T>*)> visit) {

		visit_node_fn = visit;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			NULL,
			p4est_search_visit,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Traverse the quadtree in a post-order fashion with a data callback function
	 * 
	 * @param visit Data callback function
	 */
	void traversePostOrder(std::function<void(T&)> visit) {

		visit_node_data_fn = visit;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			NULL,
			p4est_search_visit_data,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Performs a merge traversal by visiting families in a post-order fashion
	 * 
	 * @param visit_leaf Leaf node callback function
	 * @param visit_family Family node callback function
	 */
	void merge(std::function<int(Node<T>*)> visit_leaf, std::function<int(Node<T>*, std::vector<Node<T>*>)> visit_family) {

		visit_node_fn = visit_leaf;
		visit_family_fn = visit_family;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			NULL,
			p4est_search_merge_split,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Performs a split traversal by visiting families in a pre-order fashion
	 * 
	 * @param visit_leaf Leaf node callback function
	 * @param visit_family Family node callback function
	 */
	void split(std::function<int(Node<T>*)> visit_leaf, std::function<int(Node<T>*, std::vector<Node<T>*>)> visit_family) {

		visit_node_fn = visit_leaf;
		visit_family_fn = visit_family;
		int skip_levels = 0;
		p4est_search_reorder(
			p4est,
			skip_levels,
			NULL,
			p4est_search_merge_split,
			NULL,
			NULL,
			NULL
		);

	}

	/**
	 * @brief Returns the root data of the quadtree
	 * 
	 * @return T&
	 */
	T& root() {
		return map["0"]->data;
	}

	/**
	 * @brief Callback provided to `p4est_search_reorder` for node traversals
	 * 
	 * Callback template for `p4est_search_local_t`.
	 * 
	 * @param p4est The forest to traverse
	 * @param which_tree The tree id under consideration
	 * @param quadrant The quadrant under consideration
	 * @param local_num Unused
	 * @param point Unused
	 * @return int 
	 */
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
				cont = quadtree.visit_node_fn(node);
			}
		}
		return cont;
	}

	/**
	 * @brief Callback provided to `p4est_search_reorder` for data traversal
	 * 
	 * Callback template for `p4est_search_local_t`.
	 * 
	 * @param p4est The forest to traverse
	 * @param which_tree The tree id under consideration
	 * @param quadrant The quadrant under consideration
	 * @param local_num Unused
	 * @param point Unused
	 * @return int 
	 */
	static int p4est_search_visit_data(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		if (node != nullptr) {
			bool owned = node->pfirst <= rank && rank <= node->plast;
			if (owned) {
				quadtree.visit_node_data_fn(node->data);
			}
		}
		return cont;
	}

	/**
	 * @brief Callback provided to `p4est_search_reorder` for merging and spliting
	 * 
	 * Callback template for `p4est_search_local_t`.
	 * 
	 * @param p4est The forest to traverse
	 * @param which_tree The tree id under consideration
	 * @param quadrant The quadrant under consideration
	 * @param local_num Rank-local index and leaf vs. non-leaf flag
	 * @param point Unused
	 * @return int 
	 */
	static int p4est_search_merge_split(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {

		// Get data from quadtree stored in user_pointer
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		bool owned = node->pfirst <= rank && rank <= node->plast;
		
		// Get node group and communicator
		MPI_Comm node_comm;
		node_comm = node->node_comm;
		int nodeRank; MPI_Comm_rank(node_comm, &nodeRank);

		if (local_num >= 0) {
			// Quadrant is a leaf; call leaf callback
			if (node != nullptr && owned) {
				cont = quadtree.visit_node_fn(node);
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
				if (quadtree.communication_policy == NodeCommunicationPolicy::BROADCAST) {

					// Iterate through children
					for (int i = 0; i < 4; i++) {
						Node<T>* child = map[node->path + std::to_string(i)];
						
						// Communicate root rank to all others in node comm
						bool am_root = child != nullptr;
						int pnode;
						int maybe_root = (am_root ? nodeRank : 0);
						MPI_Allreduce(&maybe_root, &pnode, 1, MPI_INT, MPI_MAX, node_comm);

						// Allocate memory on my rank to store incoming node data
						if (child == nullptr) {
							// child = quadtree.node_factory->createChildNode(node, i, node->pfirst, node->plast);
							child = new Node<T>(node->getComm());
						}

						// Broadcast node
						MPI::broadcast(*child, pnode, node_comm);
						
						// Store child in children
						children[i] = child;
					}
				}
				else if (quadtree.communication_policy == NodeCommunicationPolicy::STRIPE) {
					// TODO
					throw std::runtime_error("Not implemented!");
				}
			}

			// If owned, call family branch callback
			if (owned) {
				cont = quadtree.visit_family_fn(node, children);
			}

		}
		return cont;

	}

};

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_