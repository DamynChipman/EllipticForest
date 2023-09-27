#ifndef QUADTREE_HPP_
#define QUADTREE_HPP_

#include <iostream>
#include <string>
#include <ostream>
#include <vector>
#include <functional>
#include <set>

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include "EllipticForestApp.hpp"
#include "MPI.hpp"
#include "P4est.hpp"
#include "QuadNode.hpp"

namespace EllipticForest {

static uint32_t MAP_CAPACITY_HINT = 256;

enum NodeCommunicationPolicy {
	BROADCAST,
	STRIPE
};

template<typename T> class Quadtree;

// template<typename T>
// static Quadtree<T>* quadtree_pointer = nullptr;

template<typename T>
class Quadtree : public MPI::MPIObject {

public:

	// using NodeMap = std::map<NodePathKey, Node<T>*>;
	using NodeMap = Kokkos::UnorderedMap<int, Node<T>>;
	using NodeSet = std::set<NodePathKey>;
	NodeMap node_map;
	NodeSet node_set;
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
		node_map(MAP_CAPACITY_HINT),
		p4est(p4est),
		rootDataPtr_(&rootData),
		nodeFactory(&nodeFactory) {

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
				auto& node_map = quadtree.node_map;
				auto& node_set = quadtree.node_set;

				// Get MPI info
				int rank;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);

				// Compute unique path
				std::string path = p4est::p4est_quadrant_path(quadrant);
				app.log("In quadrant: %s", path.c_str());
				// node_set.insert(path);

				// Check if quadrant is owned by this rank
				bool owned = pfirst <= rank && rank <= plast;

				// If owned, create Node in node_map
				if (owned) {
					// Create node and metadata
					// Node<T>* node = new Node<T>;
					Node<T> node;

					// Create node's data
					if (quadrant->level == 0) {
						// Quadrant is root; init with root data
						node = quadtree.nodeFactory->createNode(*quadtree.rootDataPtr_, path, quadrant->level, pfirst, plast);
					}
					else {
						// Quadrant is not root; init from parent and sibling index
						Node<T> parentNode = quadtree.getParentFromPath(path);
						int siblingID = p4est_quadrant_child_id(quadrant);
						node = quadtree.nodeFactory->createChildNode(parentNode, siblingID, pfirst, plast);
					}

					// Leaf flag
					node.leaf = local_num >= 0;

					// Put in node_map
					// node_map[path] = node;
					
					// int index = quadtree.nodeIndex(path);
					app.log("  Inserting node with path: %s", path.c_str());
					// node_map.insert(index, node);
					quadtree.insert(path, node);
				}
				// else {
				// 	// Not owned by local rank, put in nullptr for this node
				// 	node_map[path] = nullptr;
				// }

				return 1;
			},
			NULL,
			NULL
		);

		// Restore p4est user pointer
		p4est->user_pointer = savedUserPointer;

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
			node_map = other.node_map;
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

	int nodeIndex(NodePathKey key) {
		int index;
		auto b = node_set.begin();
		auto p = node_set.find(key);
		if (p == node_set.end()) {
			index =  -1;
		}
		else {
			index = std::distance(b, p);
		}
		return index;
	}

	void insert(NodePathKey key, Node<T>& node) {
		// 
		node_set.insert(key);

		// 
		int index;
		auto b = node_set.begin();
		auto p = node_set.find(key);
		if (p == node_set.end()) {
			index =  -1;
		}
		else {
			index = std::distance(b, p);
		}

		// 
		node_map.insert(index, node);

	}

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
			p4est_search_split,
			NULL,
			NULL,
			NULL
		);

	}

	Node<T> getParentFromPath(std::string path) {
		std::string parent_path = path.substr(0, path.length() - 1);
		auto index = std::distance(node_set.begin(), node_set.find(parent_path));
		auto i = node_map.find(index);
		return node_map.value_at(i);
	}

	T& root() {
		return node_map["0"]->data;
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
		// app.log("Post-quadrant callback. " + node->str());

		// printf("[RANK %i / %i]\n", rank, ranks);
		// std::cout << "\tlocal_num = " << local_num << std::endl;
		// std::cout << "\tnode = " << node << std::endl;

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
						// app.log("CALL MPI_Allreduce");
						MPI_Allreduce(&maybeRoot, &pnode, 1, MPI_INT, MPI_MAX, nodeComm);

						// Allocate memory on my rank to store incoming node data
						if (child == nullptr) {
							child = new Node<T>(node->getComm());
						}

						// Broadcast node
						// app.log("Broadcasting child node %i, root = %i", i, pnode);
						MPI::broadcast(*child, pnode, nodeComm);
						
						// Store child in children
						children[i] = child;
					}

					node->freeMPIGroupComm(&nodeGroup, &nodeComm);
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

#if 0

/**
 * @brief Data structure for a full quadtree with adaptivity features
 * 
 * @tparam T Datatype of each node
 */
template<typename T>
class Quadtree : public MPIObject {

public:

    typedef std::vector<std::vector<int>> LevelArray;

protected:

	/**
	 * @brief Storage for p4est data structure
	 * 
	 */
	p4est_t* p4est_;

	/**
	 * @brief LevelArray with global IDs
	 * 
	 */
    LevelArray globalIndices_{};

	/**
	 * @brief LevelArray with parent IDs
	 * 
	 */
    LevelArray parentIndices_{};

	/**
	 * @brief LevelArray with first child IDs
	 * 
	 */
    LevelArray childIndices_{};

	LevelArray leafIndices_{};

	int globalCounter = 0;

	/**
	 * @brief Storage for node data
	 * 
	 */
    std::vector<T> data_{};

	/**
	 * @brief Wrapper of quadtree data for passing to p4est functions
	 * 
	 */
    typedef struct quadtree_data_wrapper {

		LevelArray* g;
		LevelArray* p;
		LevelArray* c;
		std::vector<T>* d;
		int global_counter;
		Quadtree<T>* quadtree;

	} quadtree_data_wrapper_t;



	/**
	 * @brief Wrapper of node and visit function for passing to p4est functions
	 * 
	 */
    typedef struct quadtree_traversal_wrapper {

		std::function<void(T&)> visit;
		T& data;

	} quadtree_traversal_wrapper_t;

public:

	/**
	 * @brief Capsulated data structure for returning a node's information
	 * 
	 */
	struct QuadtreeNode {
		T* data;
		int level;
		int levelID;
		int globalID;
		int parentID;
		std::vector<int> childrenIDs = {-1, -1, -1, -1};
		int leafID;
	};

    Quadtree() :
		MPIObject(MPI_COMM_WORLD), p4est_(nullptr)
			{}
	
	Quadtree(p4est_t* p4est) :
		MPIObject(MPI_COMM_WORLD), p4est_(p4est) 
			{}

	LevelArray& globalIndices() { return globalIndices_; }
	const LevelArray& globalIndices() const { return globalIndices_; }

	LevelArray& parentIndices() { return parentIndices_; }
	const LevelArray& parentIndices() const { return parentIndices_; }

	LevelArray& childIndices() { return childIndices_; }
	const LevelArray& childIndices() const { return childIndices_; }

	LevelArray& leafIndices() { return leafIndices_; }
	const LevelArray& leafIndices() const { return leafIndices_; }

	T& root() { return data_[0]; }
	const T& root() const { return data_[0]; }

	std::vector<T>& data() { return data_; }
	const std::vector<T>& data() const { return data_; }

	/**
	 * @brief Utility function that gets node data given the global ID
	 * 
	 * Iterates over the entire tree until the node is found; non-optimal performance
	 * 
	 * @param ID Global ID of desired node
	 * @return QuadtreeNode 
	 */
	QuadtreeNode getNode(int ID) {

		QuadtreeNode node;

		// Find node in global level array
		bool breakLoop = false;
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] == ID) {
					// Node found, get data and return
					// node.level = l;
					// node.levelID = i;
					// node.globalID = globalIndices_[l][i];
					// node.parentID = parentIndices_[l][i];
					// node.childrenIDs[0] = childIndices_[l][i];
					// if (node.childrenIDs[0] != -1) {
					// 	auto it = std::find(globalIndices_[l+1].begin(), globalIndices_[l+1].end(), node.childrenIDs[0]);
					// 	int c_idx = it - globalIndices_[l+1].begin();
					// 	node.childrenIDs[1] = globalIndices_[l+1][c_idx+1];
					// 	node.childrenIDs[2] = globalIndices_[l+1][c_idx+2];
					// 	node.childrenIDs[3] = globalIndices_[l+1][c_idx+3];
					// }
					// node.leafID = leafIndices_[l][i];
					// node.data = &data_[node.globalID];
					// return node;
					return getNode(l, i);
				}
			}
		}

	}

	QuadtreeNode getNode(int level, int idx) {
		QuadtreeNode node;
		node.level = level;
		node.levelID = idx;
		node.globalID = globalIndices_[level][idx];
		node.parentID = parentIndices_[level][idx];
		node.childrenIDs[0] = childIndices_[level][idx];
		if (node.childrenIDs[0] != -1) {
			auto it = std::find(globalIndices_[level+1].begin(), globalIndices_[level+1].end(), node.childrenIDs[0]);
			int c_idx = it - globalIndices_[level+1].begin();
			node.childrenIDs[1] = globalIndices_[level+1][c_idx+1];
			node.childrenIDs[2] = globalIndices_[level+1][c_idx+2];
			node.childrenIDs[3] = globalIndices_[level+1][c_idx+3];
		}
		node.leafID = leafIndices_[level][idx];
		node.data = &data_[node.globalID];
		return node;
	}

	/**
	 * @brief Builds the quadtree from a p4est data structure
	 * 
	 * @param p4est p4est instance
	 * @param rootData Copy of root data
	 * @param initDataFunction Function that initializes a child node from a parent and the child index [0..3]
	 */
	void buildFromP4est(p4est_t* p4est, T rootData, std::function<T(T& parentNode, int childIndex)> initDataFunction) {
		p4est_ = p4est;
		buildLevelArrays_();
		buildData_(rootData, initDataFunction);
	}

	/**
	 * @brief Builds the quadtree from the root data; quadtree can be refined and coarsened using `refineNode` and `coarsenNode`
	 * 
	 * @sa refineNode
	 * @sa coarsenNode
	 * 
	 * @param rootData Copy of root data 
	 */
	void buildFromRoot(T rootData) {
		data_.push_back(rootData);
		globalIndices_.push_back({0});
		parentIndices_.push_back({-1});
		childIndices_.push_back({-1});
		leafIndices_.push_back({0});
	}

	/**
	 * @brief Traverses the quadtree in pre-order fashion
	 * 
	 * @param visit Function that is called during pre-order traversal with reference to node data
	 */
	void traversePreOrder(std::function<void(T&)> visit) {

		for (auto& d : data_) visit(d);

	}

	void traversePreOrder(std::function<bool(QuadtreeNode node)> visit) {

		_traversePreOrder(visit, 0, 0, true);

		// QuadtreeNode node;
		// for (int l = 0; l < globalIndices_.size(); l++) {
		// 	for (int i = 0; i < globalIndices_[l].size(); i++) {
		// 		node.level = l;
		// 		node.levelID = i;
		// 		node.globalID = globalIndices_[l][i];
		// 		node.parentID = parentIndices_[l][i];
		// 		node.childrenIDs[0] = childIndices_[l][i];
		// 		if (node.childrenIDs[0] != -1) {
		// 			auto it = std::find(globalIndices_[l+1].begin(), globalIndices_[l+1].end(), node.childrenIDs[0]);
		// 			int c_idx = it - globalIndices_[l+1].begin();
		// 			node.childrenIDs[1] = globalIndices_[l+1][c_idx+1];
		// 			node.childrenIDs[2] = globalIndices_[l+1][c_idx+2];
		// 			node.childrenIDs[3] = globalIndices_[l+1][c_idx+3];
		// 		}
		// 		node.data = &data_[node.globalID];
		// 		visit(node);
		// 	}
		// }

	}

	/**
	 * @brief Traverses the quadtree in post-order fashion
	 * 
	 * @param visit Function that is called during post-order traversal with reference to node data
	 */
	void traversePostOrder(std::function<void(T&)> visit) {

		_traversePostOrder(visit, 0, 0);

	}

	bool isValid() {

		bool dataGood = false;
		bool nodesGood = false;

		// Get total number of nodes stored via global level array
		int nNodes = 0;
		for (auto& la : globalIndices_) {
			nNodes += la.size();
		}

		// Check if number of nodes is equal to number of data entries
		dataGood = nNodes == data_.size();

		// Traverse tree via post-order and check if all nodes are visited
		int nCounter = 0;
		this->traversePostOrder([&](T& nodeData){
			nCounter++;
		});
		nodesGood = nNodes == nCounter;

		return dataGood && nodesGood;

	}

	/**
	 * @brief Does the merge algorithm detailed in reference paper
	 * 
	 * Starting at the lowest level, calls the `visit` function for the parent and children, and
	 * iterates up the tree, similar to a post-order traversal
	 * 
	 * The `visit` function is called with the following data order:
	 * 		visit(parentData, child0Data, child1Data, child2Data, child3Data)
	 * 
	 * @param visit Function that is called for parent and child data with references to parent and child data
	 */
	void merge(std::function<void(T&, T&, T&, T&, T&)> visit) {

		for (int l = globalIndices_.size()-1; l > 0; l--) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (i % 4 == 0) {
					int c0_idx = i;
					int c1_idx = i+1;
					int c2_idx = i+2;
					int c3_idx = i+3;

					int c0ID = globalIndices_[l][c0_idx];
					int c1ID = globalIndices_[l][c1_idx];
					int c2ID = globalIndices_[l][c2_idx];
					int c3ID = globalIndices_[l][c3_idx];
					int pID = parentIndices_[l][i];

					visit(
						data_[pID],
						data_[c0ID],
						data_[c1ID],
						data_[c2ID],
						data_[c3ID]
					);
				}
			}
		}

	}

	/**
	 * @brief Does the split algorithm detailed in reference paper
	 * 
	 * Starting at the root, calls the `visit` function for the parent and children, and
	 * iteraters down the tree, similiar to a pre-order traversal
	 * 
	 * The `visit` function is called with the following data order:
	 * 		visit(parentData, child0Data, child1Data, child2Data, child3Data)
	 * 
	 * @param visit Function that is called for parent and child data with references to parent and child data
	 */
	void split(std::function<void(T&, T&, T&, T&, T&)> visit) {

		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {

				int pID = globalIndices_[l][i];
				int c0ID = childIndices_[l][i];

				if (c0ID != -1) {
					auto it = std::find(globalIndices_[l+1].begin(), globalIndices_[l+1].end(), c0ID);
					int c_idx = it - globalIndices_[l+1].begin();
					int c1ID = globalIndices_[l+1][c_idx+1];
					int c2ID = globalIndices_[l+1][c_idx+2];
					int c3ID = globalIndices_[l+1][c_idx+3];

					visit(
						data_[pID],
						data_[c0ID],
						data_[c1ID],
						data_[c2ID],
						data_[c3ID]
					);
				}
			}
		}

	}

	/**
	 * @brief Refines a given node, creating four children
	 * 
	 * @param nodeID Global node ID of node to refine
	 * @param parent2childrenFunction Function that creates four children data from parent data
	 */
	void refineNode(int nodeID, std::function<std::vector<T>(T&)> parent2childrenFunction) {

		// Get node information
		QuadtreeNode node = getNode(nodeID);

		if (node.childrenIDs[0] != -1) {
			std::string errorMessage = "[EllipticForest::Quadtree::refineNode] Selected node to refine is not a leaf node; cannot refine.";
			std::cerr << errorMessage << std::endl;
			throw std::invalid_argument(errorMessage);
		}

		// Create new fine nodes (children)
		T& p = data_[node.globalID];
		std::vector<T> newNodeData = parent2childrenFunction(p);

		// Iterate through tree via level arrays and update all nodes
		LevelArray newGlobalIndices = globalIndices_;
		LevelArray newParentIndices = parentIndices_;
		LevelArray newChildIndices = childIndices_;
		LevelArray newLeafIndices = leafIndices_;
		std::vector<T> newData(data_.size() + 4);
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] > nodeID) {
					newGlobalIndices[l][i] = globalIndices_[l][i] + 4;
					if (newLeafIndices[l][i] != -1) {
						newLeafIndices[l][i] = leafIndices_[l][i] + 3;
					}
				}
				if (parentIndices_[l][i] > nodeID) {
					newParentIndices[l][i] = parentIndices_[l][i] + 4;
				}
				if (childIndices_[l][i] > nodeID) {
					newChildIndices[l][i] = childIndices_[l][i] + 4;
				}
				if (globalIndices_[l][i] != nodeID) {
					newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
				}
			}
		}

		// Insert new nodes and data into tree
		int l = node.level;
		int i = node.levelID;
		newChildIndices[l][i] = nodeID + 1;
		newLeafIndices[l][i] = -1;
					
		std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
		std::vector<int> childrenPIDS(4, nodeID);
		std::vector<int> childrenCIDS(4, -1);
		std::vector<int> childrenLIDS = {node.leafID + 0, node.leafID + 1, node.leafID + 2, node.leafID + 3};

		if (l == globalIndices_.size()-1) {
			// New level needs to be created
			newGlobalIndices.push_back(childrenGIDS);
			newParentIndices.push_back(childrenPIDS);
			newChildIndices.push_back(childrenCIDS);
			newLeafIndices.push_back(childrenLIDS);
		}
		else {
			// Nodes inserted before ones on same level
			int nextLevelFirstChildID = -1;
			int j = 1;
			while (nextLevelFirstChildID == -1) {
				nextLevelFirstChildID = childIndices_[l][i+j];
				j++;
			}
			QuadtreeNode nextLevelFirstChildNode = getNode(nextLevelFirstChildID);

			std::vector<int>::iterator iterG = std::find(newGlobalIndices[l+1].begin(), newGlobalIndices[l+1].end(), nextLevelFirstChildNode.globalID+4);
			std::vector<int>::iterator iterP = std::find(newParentIndices[l+1].begin(), newParentIndices[l+1].end(), nextLevelFirstChildNode.parentID+4);
			std::vector<int>::iterator iterC = std::find(newChildIndices[l+1].begin(), newChildIndices[l+1].end(), nextLevelFirstChildNode.childrenIDs[0]);
			std::vector<int>::iterator iterL = std::find(newLeafIndices[l+1].begin(), newLeafIndices[l+1].end(), nextLevelFirstChildNode.leafID+3);
			
			newGlobalIndices[l+1].insert(iterG, childrenGIDS.begin(), childrenGIDS.end());
			newParentIndices[l+1].insert(iterP, childrenPIDS.begin(), childrenPIDS.end());
			newChildIndices[l+1].insert(iterC, childrenCIDS.begin(), childrenCIDS.end());
			newLeafIndices[l+1].insert(iterL, childrenLIDS.begin(), childrenLIDS.end());
		}

		newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
		for (int j = 0; j < 4; j++) {
			newData[nodeID + j + 1] = newNodeData[j];
		}

		// Reset quadtree data
		globalIndices_ = newGlobalIndices;
		parentIndices_ = newParentIndices;
		childIndices_ = newChildIndices;
		leafIndices_ = newLeafIndices;
		data_ = newData;

		return;

	}

	/**
	 * @brief Coarsens a given node, deleting the children
	 * 
	 * @param nodeID Global node ID of node to coarsen
	 * @param children2parentFunction Function that creates parent data given the four children data
	 */
	void coarsenNode(int nodeID, std::function<T(T&, T&, T&, T&)> children2parentFunction) {

		// Get node information
		QuadtreeNode node = getNode(nodeID);
		QuadtreeNode firstChildNode = getNode(node.childrenIDs[0]);

		// Create new coarse node (parent)
		T& c0 = data_[node.childrenIDs[0]];
		T& c1 = data_[node.childrenIDs[1]];
		T& c2 = data_[node.childrenIDs[2]];
		T& c3 = data_[node.childrenIDs[3]];
		T newNodeData = children2parentFunction(c0, c1, c2, c3);

		// Update leaf ID of node to coarse with first child's leaf ID
		leafIndices_[node.level][node.levelID] = firstChildNode.leafID;

		// Erase child nodes from level arrays
		int l = firstChildNode.level;
		int i = firstChildNode.levelID;
		globalIndices_[l].erase(globalIndices_[l].begin() + i, globalIndices_[l].begin() + i + 4);
		parentIndices_[l].erase(parentIndices_[l].begin() + i, parentIndices_[l].begin() + i + 4);
		childIndices_[l].erase(childIndices_[l].begin() + i, childIndices_[l].begin() + i + 4);
		leafIndices_[l].erase(leafIndices_[l].begin() + i, leafIndices_[l].begin() + i + 4);

		// Iterate through tree via level arrays and create new ones
		LevelArray newGlobalIndices = globalIndices_;
		LevelArray newParentIndices = parentIndices_;
		LevelArray newChildIndices = childIndices_;
		LevelArray newLeafIndices = leafIndices_;
		std::vector<T> newData(data_.size() - 4);
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] > nodeID) {
					newGlobalIndices[l][i] = globalIndices_[l][i] - 4;
					if (newLeafIndices[l][i] != -1) {
						newLeafIndices[l][i] = leafIndices_[l][i] - 3;
					}
				}
				if (parentIndices_[l][i] > nodeID) {
					newParentIndices[l][i] = parentIndices_[l][i] - 4;
				}
				if (childIndices_[l][i] > nodeID) {
					newChildIndices[l][i] = childIndices_[l][i] - 4;
				}
				if (globalIndices_[l][i] != nodeID) {
					newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
				}
				else {
					newChildIndices[l][i] = -1;
					newData[newGlobalIndices[l][i]] = newNodeData;
				}
			}
		}

		// Reset quadtree data
		globalIndices_ = newGlobalIndices;
		parentIndices_ = newParentIndices;
		childIndices_ = newChildIndices;
		leafIndices_ = newLeafIndices;
		data_ = newData;

		return;

	}

	std::string str() const {

		std::string out = "";
		const auto& G = globalIndices_;
		const auto& P = parentIndices_;
		const auto& C = childIndices_;
		const auto& L = leafIndices_;

		out += "Global Indices:\n";
		for (int l = 0; l < G.size(); l++) {
			out += "l = " + std::to_string(l) + ": [";
			for (int i = 0; i < G[l].size(); i++) {
				out += std::to_string(G[l][i]) + ", ";
			}
			out += "]\n";
		}
		out += "Parent Indices:\n";
		for (int l = 0; l < P.size(); l++) {
			out += "l = " + std::to_string(l) + ": [";
			for (int i = 0; i < P[l].size(); i++) {
				out += std::to_string(P[l][i]) + ", ";
			}
			out += "]\n";
		}
		out += "Child Indices:\n";
		for (int l = 0; l < C.size(); l++) {
			out += "l = " + std::to_string(l) + ": [";
			for (int i = 0; i < C[l].size(); i++) {
				out += std::to_string(C[l][i]) + ", ";
			}
			out += "]\n";
		}
		out += "Leaf Indices:\n";
		for (int l = 0; l < L.size(); l++) {
			out += "l = " + std::to_string(l) + ": [";
			for (int i = 0; i < L[l].size(); i++) {
				out += std::to_string(L[l][i]) + ", ";
			}
			out += "]\n";
		}

		return out;

	}

	/**
	 * @brief Outstream operator for Quadtree<T>
	 * 
	 * @param os ostream reference
	 * @param quadtree Quadtree reference
	 * @return std::ostream& 
	 */
	friend std::ostream& operator<<(std::ostream& os, const Quadtree& quadtree) {

		const auto& G = quadtree.globalIndices();
		const auto& P = quadtree.parentIndices();
		const auto& C = quadtree.childIndices();
		const auto& L = quadtree.leafIndices();

		os << "Global Indices:" << std::endl;
		for (int l = 0; l < G.size(); l++) {
			os << "l = " << l << ": [";
			for (int i = 0; i < G[l].size(); i++) {
				os << G[l][i] << ", ";
			}
			os << "]\n";
		}
		os << "Parent Indices:" << std::endl;
		for (int l = 0; l < P.size(); l++) {
			os << "l = " << l << ": [";
			for (int i = 0; i < P[l].size(); i++) {
				os << P[l][i] << ", ";
			}
			os << "]\n";
		}
		os << "Child Indices:" << std::endl;
		for (int l = 0; l < C.size(); l++) {
			os << "l = " << l << ": [";
			for (int i = 0; i < C[l].size(); i++) {
				os << C[l][i] << ", ";
			}
			os << "]\n";
		}
		os << "Leaf Indices:" << std::endl;
		for (int l = 0; l < L.size(); l++) {
			os << "l = " << l << ": [";
			for (int i = 0; i < L[l].size(); i++) {
				os << L[l][i] << ", ";
			}
			os << "]\n";
		}

		return os;

	}

protected:    

	/**
	 * @brief p4est callback function for the pre-visit callback
	 * 
	 * @param p4est p4est instance
	 * @param which_tree Tree ID
	 * @param quadrant p4est quadrant
	 * @param local_num 
	 * @param point 
	 * @return int 
	 */
    static int p4est_visit_pre(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		// Write code that checks which processors have `quadrant`
		// p4est_comm_find_owner to find owner of lower left corner
		// p4est_last_descendant
		// find owner of last descedant
		// Range of ranks
		// Check if process has empty quad
		// p4est_comm_is_empty
		// Not on leaves
		// local_num != -1

		// Get access to level arrays
		quadtree_data_wrapper_t* wrapper = (quadtree_data_wrapper_t*) p4est->user_pointer;
		LevelArray& globalIndices = *(wrapper->g);
		LevelArray& parentIndices = *(wrapper->p);
		LevelArray& childIndices = *(wrapper->c);
		std::vector<T> data = *(wrapper->d);
		Quadtree<T>* quadtree = wrapper->quadtree;

		double vxyz[3];
		p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
		printf("[RANK %i] quad.x = %8.4f, quad.y = %8.4f, quad.l = %i\n", quadtree->getRank(), vxyz[0], vxyz[1], quadrant->level);

		p4est_tree_t* this_tree = p4est_tree_array_index(p4est->trees, which_tree);
		p4est_quadrant_t ld = this_tree->last_desc;
		int owner_left = p4est_comm_find_owner(p4est, which_tree, quadrant, 0);
		int owner_right = p4est_comm_find_owner(p4est, which_tree, &ld, 0);

		// printf("[RANK %i] owner_left = %i, owner_right = %i\n", quadtree->getRank(), owner_left, owner_right);

		

		// Populate global index array
		globalIndices[quadrant->level].push_back(quadtree.globalCounter++);

		// Populate parent index array
		int pID;
		if (quadrant->level == 0) {
			pID = -1;
		}
		else {
			pID = globalIndices[quadrant->level-1][globalIndices[quadrant->level-1].size() - 1];
		}
		parentIndices[quadrant->level].push_back(pID);

		return 1;
	}

	/**
	 * @brief p4est callback function for the post-visit callback
	 * 
	 * @param p4est p4est instance
	 * @param which_tree Tree ID
	 * @param quadrant p4est quadrant
	 * @param local_num 
	 * @param point 
	 * @return int 
	 */
	static int p4est_visit_post(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		
		// Get access to level arrays
		quadtree_data_wrapper_t* wrapper = (quadtree_data_wrapper_t*) p4est->user_pointer;
		LevelArray& globalIndices = *(wrapper->g);
		LevelArray& parentIndices = *(wrapper->p);
		LevelArray& childIndices = *(wrapper->c);
		std::vector<T> data = *(wrapper->d);
		Quadtree<T>* quadtree = wrapper->quadtree;

		// printf("[RANK %i] POST: global_counter = %i\n", quadtree->getRank(), wrapper->global_counter);

		// Populate child array
		int cID;
		if (local_num < 0) {
			// Current patch is not a leaf
			cID = globalIndices[quadrant->level+1][globalIndices[quadrant->level+1].size()-4];
		}
		else {
			// Patch is a leaf
			cID = -1;
		}
		childIndices[quadrant->level].push_back(cID);

		return 1;
	}

private:

	/**
	 * @brief Bulds the level arrays
	 * 
	 */
    void buildLevelArrays_() {
        
        p4est_tree_t* p4est_tree = p4est_tree_array_index(p4est_->trees, 0); // TODO: What if this is part of a forest of trees?
        std::size_t nLevels = p4est_tree->maxlevel + 1;
        globalIndices_.reserve(nLevels);
        parentIndices_.reserve(nLevels);
        childIndices_.reserve(nLevels);
		leafIndices_.reserve(nLevels);

        for (auto l = 0; l < nLevels; l++) {
            globalIndices_.push_back(std::vector<int>{});
            parentIndices_.push_back(std::vector<int>{});
            childIndices_.push_back(std::vector<int>{});
			leafIndices_.push_back(std::vector<int>{});

            globalIndices_[l].reserve((std::size_t) pow(2, 2*l));
            parentIndices_[l].reserve((std::size_t) pow(2, 2*l));
            childIndices_[l].reserve((std::size_t) pow(2, 2*l));
			leafIndices_[l].reserve((std::size_t) pow(2, 2*l));
        }

        // Set up user pointer
        void* userPointerSaved = p4est_->user_pointer;

        // Create wrapper and store in p4est user pointer
        quadtree_data_wrapper_t wrapper;
        wrapper.g = &globalIndices_;
		wrapper.p = &parentIndices_;
		wrapper.c = &childIndices_;
		wrapper.d = &data_;
		wrapper.global_counter = 0;
		wrapper.quadtree = this;
		p4est_->user_pointer = &wrapper;

        // Call `p4est_search_reorder` to traverse tree and populate level arrays
		int skipLevels = 0;
		p4est_search_reorder(p4est_, skipLevels, NULL, this->p4est_visit_pre, this->p4est_visit_post, NULL, NULL);

		// Restore original p4est user pointer
		p4est_->user_pointer = userPointerSaved;

    }

	/**
	 * @brief Builds the data
	 * 
	 * @param rootData Reference of data at root of tree
	 * @param initDataFunction Function that creates a child given the parent data and child index [0..3]
	 */
    void buildData_(T& rootData, std::function<T(T& parentNode, std::size_t childIndex)> initDataFunction) {

		// Count total number of nodes
		int nodeCounter = 0;
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				nodeCounter++;
			}
		}
		data_.resize(nodeCounter); // Assumes a default constructor is built for T

		// Iterate through tree via levels
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				// Call init function and place in data array
				if (l == 0) {
					data_[globalIndices_[l][i]] = rootData;
				}
				else {
					// Get parent data
					int pID = parentIndices_[l][i];
					T& parentData = data_[pID];

					std::size_t childIndex = i % 4;
					data_[globalIndices_[l][i]] = initDataFunction(parentData, childIndex);
				}
			}
		}

	}

	/**
	 * @brief Recursive call to traverse post order
	 * 
	 * @param visit Visit function
	 * @param level Level
	 * @param idx Level index
	 */
	void _traversePostOrder(std::function<void(T&)> visit, int level, int idx) {
		int gID = globalIndices_[level][idx];
		int cID = childIndices_[level][idx];
		if (cID == -1) {
			visit(data_[gID]);
			return;
		}
		else {
			auto it = std::find(globalIndices_[level+1].begin(), globalIndices_[level+1].end(), cID);
			int c_idx = it - globalIndices_[level+1].begin();
			for (int i = 0; i < 4; i++) {
				_traversePostOrder(visit, level+1, c_idx+i);
			}
			visit(data_[gID]);
		}
	}

	bool _traversePreOrder(std::function<bool(QuadtreeNode)> visit, int level, int idx, bool cont) {

		bool continueTraversal;
		if (cont) {
			// Get node by level and index
			QuadtreeNode node = getNode(level, idx);

			// Visit node
			cont = visit(node);

			// Visit children
			if (cont) {
				if (node.childrenIDs[0] != -1) {
					auto it = std::find(globalIndices_[level+1].begin(), globalIndices_[level+1].end(), node.childrenIDs[0]);
					int c_idx = it - globalIndices_[level+1].begin();
					for (int i = 0; i < 4; i++) {
						cont = _traversePreOrder(visit, level+1, c_idx+i, cont);
					}
				}
			}
			else {
				return false;
			}
		}
		else {
			return false;
		}

		return cont;

	}

};

#endif

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_