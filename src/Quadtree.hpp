/**
 * @file Quadtree.hpp
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Path-indexed quadtree data structure
 */
#ifndef QUADTREE_HPP_
#define QUADTREE_HPP_

#include <cmath>
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

enum BalancePolicy {
	SELF,
	FACE,
	CORNER,
	FULL
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
	 * @brief Sibling (group of 4) nodes callback function for use in p4est routines
	 * 
	 */
	std::function<int(std::vector<Node<T>*>)> visit_siblings_fn;

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
	 * @brief Node family callback function for use in propagating
	 * 
	 */
	std::function<int(Node<T>*, std::vector<Node<T>*>)> propagate_fn;

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

	Quadtree(MPI::Communicator comm, T& root_data, AbstractNodeFactory<T>& node_factory, std::vector<double> extent = {0, 1, 0, 1}) :
		MPIObject(comm),
		root_data_ptr_(&root_data),
		node_factory(&node_factory) {

		// Create p4est instance
		p4est_connectivity_t* conn = p4est::p4est_connectivity_new_square_domain(extent[0], extent[1], extent[2], extent[3]);
		p4est = p4est_new_ext(this->getComm(), conn, 0, 0, true, 0, nullptr, nullptr);
		p4est->user_pointer = this;

		// Store root data in map
		std::string root_path = "0";
		map[root_path] = this->node_factory->createNode(root_data, root_path, 0, 0, this->getSize()-1);
		map[root_path]->leaf = true;

	}
	
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
						int sibling_id = p4est_quadrant_child_id(quadrant);
						node = quadtree.node_factory->createChildNode(parent_node, sibling_id, pfirst, plast);
					}

					// Additional info for node
					node->leaf = local_num >= 0;
					node->node_comm = node_comm;

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

	Quadtree(Quadtree& other) :
		MPIObject(other.getComm()),
		p4est(other.p4est),
		root_data_ptr_(&(other.root())),
		node_factory(other.node_factory) {

		//
		map = other.map;
		for (typename NodeMap::iterator iter = map.begin(); iter != map.end(); ++iter) {
			iter->second = new Node<T>(*other.map[iter->first]);
		}

	}

	/**
	 * @brief Destroy the Quadtree object; deletes allocated map data
	 * 
	 */
	~Quadtree() {
		// Iterate through map and delete any owned nodes
		for (typename NodeMap::iterator iter = map.begin(); iter != map.end(); ++iter) {
			if (iter->second != nullptr) {
				delete iter->second;
			}
		}
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
	 * @brief Refine the quadtree according to a refinement criteria
	 * 
	 * This is the preferred interface for quadtree refining/coarsening as this just wraps p4est
	 * routines. If p4est is handled by an external library, use the `refineNode` function.
	 * 
	 * Note that all initialization of new nodes is handled by the node factory.
	 * 
	 * @sa coarsen
	 * @sa refineNode
	 * @sa coarsenNode
	 * 
	 * @param refine_recursive Flag to specify a recursive refinement
	 * @param refine_function Function that returns if the provided leaf node should be refined
	 */
	void refine(bool refine_recursive, std::function<int(Node<T>*)> refine_function, std::function<int(Node<T>*, std::vector<Node<T>*>)> propagate_function=nullptr) {

		visit_node_fn = refine_function;
		propagate_fn = propagate_function;
		p4est_refine(
			p4est,
			(int) refine_recursive,
			p4est_refine_callback,
			p4est_init_refined_callback
		);

	}

	/**
	 * @brief Refine a specific node in the quadtree
	 * 
	 * This interface for refining/coarsening is for refining specific nodes or for when p4est is
	 * handled by an external library. When the manipulation of the p4est quadtree (leaf-indexed 
	 * quadtree) is managed by another library, then this function should be called with
	 * `change_p4est=false` at each callback stage in the refining/coarsening.
	 * 
	 * Only leaf nodes may be refined this way.
	 * 
	 * Note that all initialization of new nodes is handled by the node factory.
	 * 
	 * @sa refine
	 * @sa coarsen
	 * @sa coarsenNode
	 * 
	 * @param path Path to node to refine
	 * @param change_p4est Flag to change p4est as well; if true, wraps `refine`
	 */
	void refineNode(std::string path, bool change_p4est=false) {

		auto* node = map[path];
		if (node != nullptr) {
			if (!node->leaf) {
				throw std::invalid_argument("Node with path = " + path + " is not a leaf node");
			}
		}

		if (change_p4est) {
			// Need to change p4est; use refine function which wraps p4est_refine
			refine(
				false,
				[&](Node<T>* node_){
					return (int) (node_->path == path);
				}
			);
		}
		else {
			// p4est already changed or handled externally, just need to change quadtree
			if (node != nullptr) {
				for (int i = 0; i < 4; i++) {
					int pfirst = node->pfirst;
					int plast = node->plast;
					std::string child_path = path + std::to_string(i);
					map[child_path] = node_factory->createChildNode(node, i, pfirst, plast);
					map[child_path]->leaf = true;
				}

			}
			else {
				// What to do when node is nullptr
			}

		}

	}

	/**
	 * @brief Coarsen the quadtree according to a coarsening criteria
	 * 
	 * This is the preferred interface for quadtree refining/coarsening as this just wraps p4est
	 * routines. If p4est is handled by an external library, use the `coarsenNode` function.
	 * 
	 * Note that all initialization of new nodes is handled by the node factory.
	 * 
	 * @sa refine
	 * @sa refineNode
	 * @sa coarsenNode
	 * 
	 * @param coarsen_recursive Flag to specify a recursive coarsening
	 * @param coarsen_function Function that returns if the sibling nodes should be coarsened
	 */
	void coarsen(bool coarsen_recursive, std::function<int(std::vector<Node<T>*>)> coarsen_function, std::function<int(Node<T>*, std::vector<Node<T>*>)> propagate_function=nullptr) {

		visit_siblings_fn = coarsen_function;
		propagate_fn = propagate_function;
		p4est_coarsen(
			p4est,
			(int) coarsen_recursive,
			p4est_coarsen_callback,
			p4est_init_coarsened_callback
		);

	}

	/**
	 * @brief Coarsen a specific node in the quadtree
	 * 
	 * This interface for refining/coarsening is for coarsening specific nodes or for when p4est is
	 * handled by an external library. When the manipulation of the p4est quadtree (leaf-indexed 
	 * quadtree) is managed by another library, then this function should be called with
	 * `change_p4est=false` at each callback stage in the refining/coarsening.
	 * 
	 * Only first-generation parents (node with all leaf children) may be coarsened this way.
	 * 
	 * Note that all initialization of new nodes is handled by the node factory.
	 * 
	 * @sa refine
	 * @sa coarsen
	 * @sa refineNode
	 * 
	 * @param path Path to node to coarsen
	 * @param change_p4est Flag to change p4est as well; if true, wraps `coarsen`
	 */
	void coarsenNode(std::string path, bool change_p4est=false) {

		auto* node = map[path];
		if (node != nullptr) {
			bool node_is_first_gen_parent = true;
			for (int i = 0; i < 4; i++) {
				std::string child_path = path + std::to_string(i);
				auto child_iter = map.find(child_path);
				bool child_exists = child_iter != map.end();
				if (!child_exists) {
					throw std::invalid_argument("Node with path = " + path + " does not have children.");
				}
				node_is_first_gen_parent = map[child_path]->leaf;
			}
			if (!node_is_first_gen_parent) {
				throw std::invalid_argument("Node with path = " + path + " is not a first-generation parent (all children nodes are leaves).");
			}
		}

		if (change_p4est) {
			coarsen(
				false,
				[&](std::vector<Node<T>*> nodes){
					std::string first_child_path = nodes[0]->path;
					std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
					return (int) (parent_path == path);
				}
			);
		}
		else {
			if (node != nullptr) {
				std::vector<std::string> children_paths(4);
				std::vector<Node<T>*> children_nodes(4);
				for (int i = 0; i < 4; i++) {
					children_paths[i] = path + std::to_string(i);
					children_nodes[i] = map[children_paths[i]];
				}
				int pfirst = children_nodes[0]->pfirst;
				int plast = children_nodes[3]->plast;
				map[path] = node_factory->createParentNode(children_nodes, pfirst, plast);
				map[path]->leaf = true;

				for (auto child_path : children_paths) {
					if (map[child_path] != nullptr) {
						delete map[child_path];
						map[child_path] = nullptr;
					}
				}
			}
		}

	}

	void balance(BalancePolicy balance_policy) {
		
		p4est_connect_type_t btype;
		switch (balance_policy) {
			case BalancePolicy::SELF:
				btype = P4EST_CONNECT_SELF;
				break;

			case BalancePolicy::FACE:
				btype = P4EST_CONNECT_FACE;
				break;

			case BalancePolicy::CORNER:
				btype = P4EST_CONNECT_CORNER;
				break;

			case BalancePolicy::FULL:
				btype = P4EST_CONNECT_FULL;
				break;
			
			default:
				break;
		}
		p4est_balance(
			p4est,
			btype,
			p4est_init_balance
		);

	}

	/**
	 * @brief WIP
	 * 
	 */
	void partition() {

		// if (this->getSize() != 1) {
		// 	throw std::runtime_error("Parallel partitioning not implemented!");
		// }

		p4est_partition(
			p4est,
			false,
			nullptr
		);

		// Create partitioned path-indexed quadtree structure
		NodeMap pmap;
		void* saved_user_pointer = p4est->user_pointer;
		p4est->user_pointer = (void*) &pmap;
		p4est_search_all(
			p4est,
			0,
			[](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point){
				auto& pmap = *((NodeMap*) p4est->user_pointer);
				auto path = p4est::p4est_quadrant_path(quadrant);
				pmap[path] = new Node<T>();
				pmap[path]->path = path;
				pmap[path]->level = quadrant->level;
				pmap[path]->pfirst = pfirst;
				pmap[path]->plast = plast;
				return 1;
			},
			nullptr,
			nullptr
		);
		p4est->user_pointer = saved_user_pointer;

		auto& app = EllipticForestApp::getInstance();
		app.log("contents of map:");
		for (auto& pair : map) {
			auto path = pair.first;
			auto* node = pair.second;
			int pfirst = node->pfirst;
			int plast = node->plast;
			int rank = this->getRank();
			bool owned = pfirst <= rank && rank <= plast;
			bool has_data = node != nullptr;
			app.log("path = " + path + ", owned = " + std::to_string(owned) + ", data = " + std::to_string(node->data) + ", ranks = [" + std::to_string(pfirst) + "-" + std::to_string(plast) + "]");
		}
		app.log("contents of pmap:");
		for (auto& pair : pmap) {
			auto path = pair.first;
			auto* node = pair.second;
			int pfirst = node->pfirst;
			int plast = node->plast;
			int rank = this->getRank();
			bool owned = pfirst <= rank && rank <= plast;
			bool has_data = node != nullptr;
			app.log("path = " + path + ", owned = " + std::to_string(owned) + ", data = " + std::to_string(node->data) + ", ranks = [" + std::to_string(pfirst) + "-" + std::to_string(plast) + "]");
		}

		// Iterate through map and send data this rank does not own
		// for (auto& pair : map) {
		// 	auto path = pair.first;
		// 	auto* node = pair.second;
		// 	auto* pnode = pmap[path];
			
		// 	bool owned = pnode->pfirst <= this->getRank() && this->getRank() <= pnode->plast;
		// 	bool has_data = node != nullptr;

		// 	// unsigned hash_index = p4est_quadrant_hash_fn(quadrant, nullptr);
		// 	// int tag = static_cast<int>(hash_index % (INT_MAX + 1ULL));

		// 	if (has_data && !owned) {
		// 		// Rank has node data but does not own it; send to ranks that do
		// 		for (auto dest = pnode->pfirst; dest <= pnode->plast; dest++) {
		// 			app.log("sending message: data = " + std::to_string(node->data) + ", tag = " + std::to_string(0) + ", dest = " + std::to_string(dest));
		// 			MPI::send(node->data, dest, 0, this->getComm());
		// 			app.log("sent!");
		// 		}	
		// 	}

		// 	// if (has_data) {
		// 	// 	if (owned) {
		// 	// 		// Rank has data and is owned; do nothing
		// 	// 		continue;
		// 	// 	}
		// 	// 	else {
		// 	// 		// Rank has node data but does not own it; send to ranks that do
		// 	// 		for (auto dest = pnode->pfirst; dest <= pnode->plast; dest++) {
		// 	// 			app.log("sending message: data = " + std::to_string(node->data) + ", tag = " + std::to_string(tag) + ", dest = " + std::to_string(dest));
		// 	// 			MPI::send(node->data, dest, tag, quadtree.getComm());
		// 	// 			app.log("sent!");
		// 	// 		}

		// 	// 		// Clean up unused data
		// 	// 		// delete node;
		// 	// 	}
		// 	// }
		// 	// else {
		// 	// 	if (owned) {
		// 	// 		// Rank does not have quad data but does own it; receive from rank that does
		// 	// 		T temp;

		// 	// 		MPI::Status status;
		// 	// 		app.log("probing for messages...");
		// 	// 		MPI::probe(MPI_ANY_SOURCE, tag, quadtree.getComm(), &status);
		// 	// 		app.log("receiving message: tag = " + std::to_string(tag) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 	// 		MPI::receive(temp, status.MPI_SOURCE, tag, quadtree.getComm(), MPI_STATUS_IGNORE);
		// 	// 		app.log("received message: data = " + std::to_string(temp) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 	// 		// MPI::ireceive(temp, MPI_ANY_SOURCE, tag, quadtree.getComm(), nullptr);
		// 	// 		node = quadtree.node_factory->createNode(temp, path, quadrant->level, pfirst, plast);
		// 	// 	}
		// 	// 	else {
		// 	// 		// Rank does not have data and does not need it; do nothing
		// 	// 		continue;
		// 	// 	}
		// 	// }

		// }

		// // Receive data into partitioned nodes
		// for (auto& pair : pmap) {
		// 	auto path = pair.first;
		// 	auto* pnode = pair.second;
			
		// 	bool owned = pnode->pfirst <= this->getRank() && this->getRank() <= pnode->plast;
		// 	bool has_data = pnode != nullptr;

		// 	if (owned && !has_data) {
		// 		T temp;
		// 		MPI::Status status;
		// 		app.log("probing for messages...");
		// 		MPI::probe(MPI_ANY_SOURCE, 0, this->getComm(), &status);
		// 		app.log("receiving message: 0 = " + std::to_string(0) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 		MPI::receive(temp, status.MPI_SOURCE, 0, this->getComm(), MPI_STATUS_IGNORE);
		// 		app.log("received message: data = " + std::to_string(temp) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 		pnode->data = temp;
		// 		// MPI::ireceive(temp, MPI_ANY_SOURCE, 0, this->getComm(), nullptr);
		// 		// pnode = node_factory->createNode(temp, path, pnode-, pfirst, plast);
		// 	}
		// }

		// app.log("====================================================================================================");
		// // Can I use p4est_search_all to iterate over the partitioned leaf-quadtree and then send/receive nodes?
		// p4est_search_all(
		// 	p4est,
		// 	0,
		// 	[](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {
				
		// 		int rank;
		// 		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		// 		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		// 		auto& map = quadtree.map;
		// 		auto path = p4est::p4est_quadrant_path(quadrant);
		// 		auto* node = map[path];

		// 		unsigned hash_index = p4est_quadrant_hash_fn(quadrant, nullptr);
		// 		int tag = static_cast<int>(hash_index % (INT_MAX + 1ULL));

		// 		bool owned = pfirst <= rank && rank <= plast;
		// 		bool has_data = node != nullptr;
		// 		auto& app = EllipticForestApp::getInstance();
		// 		app.log("path = " + path + ", index = " + std::to_string(tag) + ", owned = " + std::to_string(owned) + ", has_data = " + std::to_string(has_data) + ", ranks = [" + std::to_string(pfirst) + "-" + std::to_string(plast) + "]");

		// 		if (has_data) {
		// 			if (owned) {
		// 				// Rank has data and is owned; do nothing
		// 				return 1;
		// 			}
		// 			else {
		// 				// Rank has quad data but does not own it; send to ranks that do
		// 				for (auto dest = pfirst; dest <= plast; dest++) {
		// 					app.log("sending message: data = " + std::to_string(node->data) + ", tag = " + std::to_string(tag) + ", dest = " + std::to_string(dest));
		// 					MPI::send(node->data, dest, tag, quadtree.getComm());
		// 					app.log("sent!");
		// 				}

		// 				// Clean up unused data
		// 				delete node;
		// 			}
		// 		}
		// 		else {
		// 			if (owned) {
		// 				// Rank does not have quad data but does own it; receive from rank that does
		// 				T temp;

		// 				MPI::Status status;
		// 				app.log("probing for messages...");
		// 				MPI::probe(MPI_ANY_SOURCE, tag, quadtree.getComm(), &status);
		// 				app.log("receiving message: tag = " + std::to_string(tag) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 				MPI::receive(temp, status.MPI_SOURCE, tag, quadtree.getComm(), MPI_STATUS_IGNORE);
		// 				app.log("received message: data = " + std::to_string(temp) + ", src = " + std::to_string(status.MPI_SOURCE));
		// 				// MPI::ireceive(temp, MPI_ANY_SOURCE, tag, quadtree.getComm(), nullptr);
		// 				node = quadtree.node_factory->createNode(temp, path, quadrant->level, pfirst, plast);
		// 			}
		// 			else {
		// 				// Rank does not have data and does not need it; do nothing
		// 				return 1;
		// 			}
		// 		}

		// 		return 1;
		// 	},
		// 	nullptr,
		// 	nullptr
		// );

	}

	void adapt(int min_level, int max_level, std::function<int(std::vector<Node<T>*>)> coarsen_function, std::function<int(Node<T>*)> refine_function, std::function<int(Node<T>*, std::vector<Node<T>*>)> propagate_function=nullptr) {

		if (min_level == max_level) {
			refine(true, refine_function, propagate_function);
			balance(BalancePolicy::CORNER);
			return;
		}
		for (int l = min_level; l < max_level; l++) {
			refine(false, refine_function, propagate_function);
			balance(BalancePolicy::CORNER);
			// partition();
		}
		for (int l = max_level; l > min_level; l--) {
			coarsen(false, coarsen_function, propagate_function);
			balance(BalancePolicy::CORNER);
			// partition();
		}
		return;

	}

	void propagate(NodePathKey node_path, std::function<int(Node<T>*, std::vector<Node<T>*>)> family_callback) {

		if (map.find(node_path) != map.end() && map[node_path] != nullptr) {
			node_path.pop_back();
			while (node_path.length() > 0) {
				auto* parent_node = map[node_path];
				std::vector<Node<T>*> children_nodes = {
					map[node_path + "0"],
					map[node_path + "1"],
					map[node_path + "2"],
					map[node_path + "3"]
				};
				if (!family_callback(parent_node, children_nodes)) {
					return;
				}
				node_path.pop_back();
			}
		}

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

	/**
	 * @brief Callback provided to `p4est_refine` for initializing newly refined nodes
	 * 
	 * @param p4est The forest
	 * @param which_tree The tree containing quadrant
	 * @param quadrant The quadrant to be initialized
	 */
	static void p4est_init_refined_callback(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto path = p4est::p4est_quadrant_path(quadrant);
		
		std::string parent_path = path.substr(0, path.length() - 1);
		Node<T>* parent_node = map[parent_path];
		int sibling_id = p4est_quadrant_child_id(quadrant);
		int pfirst = parent_node->pfirst;
		int plast = parent_node->plast;
		map[path] = quadtree.node_factory->createChildNode(parent_node, sibling_id, pfirst, plast);
		map[path]->leaf = true;
		parent_node->leaf = false;

		if (quadtree.propagate_fn && path.back() == '3') {
			quadtree.propagate(path, quadtree.propagate_fn);
		}

	}

	/**
	 * @brief Callback provided to `p4est_coarsen` for initializing newly coarsened nodes
	 * 
	 * @param p4est The forest
	 * @param which_tree The tree containing quadrant
	 * @param quadrant The quadrant to be initialized
	 */
	static void p4est_init_coarsened_callback(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto path = p4est::p4est_quadrant_path(quadrant);

		std::vector<std::string> children_paths(4);
		std::vector<Node<T>*> children_nodes(4);
		for (int i = 0; i < 4; i++) {
			children_paths[i] = path + std::to_string(i);
			children_nodes[i] = map[children_paths[i]];
		}
		int pfirst = children_nodes[0]->pfirst;
		int plast = children_nodes[3]->plast;
		map[path] = quadtree.node_factory->createParentNode(children_nodes, pfirst, plast);
		map[path]->leaf = true;

		for (auto child_path : children_paths) {
			if (map[child_path] != nullptr) {
				delete map[child_path];
				auto it = map.find(child_path);
				map.erase(it);
				// map[child_path] = nullptr;
			}
		}

		if (quadtree.propagate_fn) {
			quadtree.propagate(path, quadtree.propagate_fn);
		}
	}

	/**
	 * @brief Callback function provided to `p4est_refine` for flagging nodes to be refined
	 * 
	 * This callback basically wraps the user provided callback for `refine`.
	 * 
	 * @param p4est The forest
	 * @param which_tree The tree containing quadrant
	 * @param quadrant The quadrant that may be refined
	 * @return int Nonzero if the quadrant shall be refined
	 */
	static int p4est_refine_callback(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto path = p4est::p4est_quadrant_path(quadrant);
		auto* node = map[path];
		return quadtree.visit_node_fn(node);
	}

	/**
	 * @brief Callback function provided to `p4est_coarsen` for flagging nodes to be coarsened
	 * 
	 * This callback basically wraps the user provided callback for `coarsen`.
	 * 
	 * @param p4est The forest
	 * @param which_tree The tree containing quadrant
	 * @param quadrants The quadrant that may be coarsened
	 * @return int Nonzero if the quadrants shall be coarsened
	 */
	static int p4est_coarsen_callback(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]) {
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		std::vector<std::string> paths(4);
		std::vector<Node<T>*> nodes(4);
		for (int i = 0; i < 4; i++) {
			auto* quadrant = quadrants[i];
			paths[i] = p4est::p4est_quadrant_path(quadrant);
			nodes[i] = map[paths[i]];
		}
		return quadtree.visit_siblings_fn(nodes);
	}

	/**
	 * @brief Callback provided to `p4est_balance` for initializing newly balanced nodes
	 * 
	 * @param p4est The forest
	 * @param which_tree The tree containing quadrant
	 * @param quadrant The quadrant to be initialized
	 */
	static void p4est_init_balance(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant) {
		auto& quadtree = *reinterpret_cast<Quadtree<T>*>(p4est->user_pointer);
		auto& map = quadtree.map;
		auto path = p4est::p4est_quadrant_path(quadrant);
		bool quad_refined = map.find(path) == map.end();
		if (quad_refined) {
			p4est_init_refined_callback(p4est, which_tree, quadrant);
		}
		else {
			p4est_init_coarsened_callback(p4est, which_tree, quadrant);
		}
	}

	static int path2Index(const NodePathKey& path, int max_length) {
		int index = 0;
		for (char c : path) {
			index += index*(4 + max_length) + (c - '0');
		}
		int max_length_bits = max_length * 2;
		int path_length_bits = ceil(log2(max_length + 1));
		// index = (path.size() << max_length_bits) | index;
		index *= std::pow(4, path.size());
		return index;
	}

};

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_