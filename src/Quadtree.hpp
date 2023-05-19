#ifndef QUADTREE_HPP_
#define QUADTREE_HPP_

#include <iostream>
#include <string>
#include <ostream>
#include <vector>
#include <functional>
#include <map>

#include "MPI.hpp"
#include "P4est.hpp"

namespace EllipticForest {

using NodePathKey = std::string;

enum QuadtreeParallelDataPolicy {
	COPY,
	STRIPE
};

template<typename T>
struct Node : public MPIObject {

	Node() :
		MPIObject(MPI_COMM_WORLD)
			{}

	Node(MPI_Comm comm) :
		MPIObject(comm)
			{}

	T data;
	std::string path;
	int level;
	int pfirst;
	int plast;

	bool isOwned() {
		return pfirst <= this->getRank() && this->getRank() <= plast;
	}

};

template<typename T>
class AbstractNodeFactory {
public:
	virtual Node<T>* createNode(T data, std::string path, int level, int pfirst, int plast) = 0;
	virtual Node<T>* createChildNode(Node<T>* parentNode, int siblingID, int pfirst, int plast) = 0;
	virtual Node<T>* createParentNode(std::vector<Node<T>*> childrenNodes, int pfirst, int plast) = 0;
};

template<typename T>
class Quadtree : public MPIObject {

public:

	using NodeMap = std::map<NodePathKey, Node<T>*>;
	NodeMap map;
	p4est_t* p4est;
	T* rootDataPtr_;
	AbstractNodeFactory<T>* nodeFactory;
	std::function<int(Node<T>*)> visitFn;
	// std::function<T(Node<T>* parentNode, int siblingID, int pfirst, int plast)> initFromParentFunction_;

	Quadtree() :
		MPIObject(MPI_COMM_WORLD),
		rootDataPtr_(nullptr),
		nodeFactory(nullptr) {}
	
	Quadtree(MPI_Comm comm, p4est_t* p4est, T rootData, AbstractNodeFactory<T>& nodeFactory) :
		MPIObject(comm),
		p4est(p4est),
		rootDataPtr_(&rootData),
		nodeFactory(&nodeFactory) {

		// Save user pointer and store self in it
		void* savedUserPointer = p4est->user_pointer;
		p4est->user_pointer = this;

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

				// Check if quadrant is owned by this rank
				bool owned = pfirst <= rank && rank <= plast;

				// If owned, create Node in map
				if (owned) {
					// Create node and metadata
					Node<T>* node = new Node<T>;
					// node->path = path;
					// node->pfirst = pfirst;
					// node->plast = plast;
					// node->level = quadrant->level;

					// Create node's data
					if (quadrant->level == 0) {
						// Quadrant is root; init with root data
						// node->data = *quadtree.rootDataPtr_;
						node = quadtree.nodeFactory->createNode(*quadtree.rootDataPtr_, path, quadrant->level, pfirst, plast);
					}
					else {
						// Quadrant is not root; init from parent and sibling index
						Node<T>* parentNode = quadtree.getParentFromPath(path);
						int siblingID = p4est_quadrant_child_id(quadrant);
						node = quadtree.nodeFactory->createChildNode(parentNode, siblingID, pfirst, plast);
						// node->data = parentNode->spawnChild(siblingID, pfirst, plast);
					}
					map[path] = node;
				}
				else {
					map[path] = nullptr;
				}

				return 1;
			},
			NULL,
			NULL
		);

	}

	static int p4est_search_visit(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
		auto& map = quadtree.map;
		auto* node = map[p4est::p4est_quadrant_path(quadrant)];
		int cont = 1;
		if (node != nullptr) {
			bool owned = node->pfirst <= rank && rank <= node->plast;
			if (owned) {
				cont = quadtree.visitFn(node);
			}
		}
		return cont;
	}

	void traversePreOrder(std::function<int(Node<T>*)> visit) {

		visitFn = visit;
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

	void traversePostOrder(std::function<int(Node<T>*)> visit) {

		visitFn = visit;
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

	Node<T>* getParentFromPath(std::string path) {
		return map[path.substr(0, path.length() - 1)];
	}

	// void initFromP4est(p4est_t* p4est, T rootData, std::function<T(Node<T>* parentNode, int siblingID, int pfirst, int plast)> initFromParentFunction) {

	// 	// Save user pointer and store self in it
	// 	void* savedUserPointer = p4est->user_pointer;
	// 	p4est->user_pointer = this;

	// 	// Use p4est_search_all to do depth first traversal and create quadtree map and nodes
	// 	int callPost = 0;
	// 	p4est_search_all(
	// 		p4est,
	// 		callPost,
	// 		[](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {

	// 			// Get quadtree
	// 			auto& quadtree = *(Quadtree<T>*) p4est->user_pointer;
	// 			auto& map = quadtree.map;

	// 			// Get MPI info
	// 			int rank;
	// 			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// 			// Compute unique path
	// 			std::string path = p4est::p4est_quadrant_path(quadrant);

	// 			// Check if quadrant is owned by this rank
	// 			bool owned = pfirst <= rank && rank <= plast;

	// 			// If owned, create Node in map
	// 			if (owned) {
	// 				// Create node and metadata
	// 				Node<T>* node = new Node<T>;
	// 				node->path = path;
	// 				node->pfirst = pfirst;
	// 				node->plast = plast;
	// 				node->level = quadrant->level;

	// 				// Create node's data
	// 				if (quadrant->level == 0) {
	// 					// Quadrant is root; init with root data
	// 					node->data = *quadtree.rootDataPtr_;
	// 				}
	// 				else {
	// 					// Quadrant is not root; init from parent and sibling index
	// 					Node<T>* parentNode = quadtree.getParent(node);
	// 					int siblingID = p4est_quadrant_child_id(quadrant);
	// 					node->data = quadtree.initFromParentFunction_(parentNode, siblingID, pfirst, plast);
	// 				}
	// 				map[path] = node;
	// 			}
	// 			else {
	// 				map[path] = nullptr;
	// 			}

	// 			return 1;
	// 		},
	// 		NULL,
	// 		NULL
	// 	);

	// }

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

	/**
	 * @brief Capsulated data structure for returning a node's information
	 * 
	 */
	struct QuadtreeNode {
		T data;
		int level;
		int levelID;
		int globalID;
		int parentID;
		std::vector<int> childrenIDs{4};
	};

public:

    Quadtree() :
		MPIObject(MPI_COMM_WORLD), p4est_(nullptr)
			{}
	
	Quadtree(p4est_t* p4est) :
		MPIObject(MPI_COMM_WORLD), p4est_(p4est) 
			{}

	LevelArray globalIndices() const { return globalIndices_; }
	LevelArray parentIndices() const { return parentIndices_; }
	LevelArray childIndices() const { return childIndices_; }
	T& root() { return data_[0]; }
	const T& root() const { return data_[0]; }
	std::vector<T>& data() { return data_; }

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
					node.level = l;
					node.levelID = i;
					node.globalID = globalIndices_[l][i];
					node.parentID = parentIndices_[l][i];
					node.childrenIDs[0] = childIndices_[l][i];
					if (node.childrenIDs[0] != -1) {
						auto it = std::find(globalIndices_[l+1].begin(), globalIndices_[l+1].end(), node.childrenIDs[0]);
						int c_idx = it - globalIndices_[l+1].begin();
						node.childrenIDs[1] = globalIndices_[l+1][c_idx+1];
						node.childrenIDs[2] = globalIndices_[l+1][c_idx+2];
						node.childrenIDs[3] = globalIndices_[l+1][c_idx+3];
					}
					node.data = data_[node.globalID];
					return node;
				}
			}
		}

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
	}

	/**
	 * @brief Traverses the quadtree in pre-order fashion
	 * 
	 * @param visit Function that is called during pre-order traversal with reference to node data
	 */
	void traversePreOrder(std::function<void(T&)> visit) {

		for (auto& d : data_) visit(d);

	}

	/**
	 * @brief Traverses the quadtree in post-order fashion
	 * 
	 * @param visit Function that is called during post-order traversal with reference to node data
	 */
	void traversePostOrder(std::function<void(T&)> visit) {

		_traversePostOrder(visit, 0, 0);

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
			return;
		}

		// Create new fine nodes (children)
		T& p = data_[node.globalID];
		std::vector<T> newNodeData = parent2childrenFunction(p);

		// Iterate through tree via level arrays and update all nodes
		LevelArray newGlobalIndices = globalIndices_;
		LevelArray newParentIndices = parentIndices_;
		LevelArray newChildIndices = childIndices_;
		std::vector<T> newData(data_.size() + 4);
		int shift = 0;
		bool nodeVisited = false;
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] > nodeID) {
					newGlobalIndices[l][i] = globalIndices_[l][i] + 4;
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
					
		std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
		std::vector<int> childrenPIDS(4, nodeID);
		std::vector<int> childrenCIDS(4, -1);

		if (l == globalIndices_.size()-1) {
			// New level needs to be created
			newGlobalIndices.push_back(childrenGIDS);
			newParentIndices.push_back(childrenPIDS);
			newChildIndices.push_back(childrenCIDS);
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
			
			newGlobalIndices[l+1].insert(iterG, childrenGIDS.begin(), childrenGIDS.end());
			newParentIndices[l+1].insert(iterP, childrenPIDS.begin(), childrenPIDS.end());
			newChildIndices[l+1].insert(iterC, childrenCIDS.begin(), childrenCIDS.end());
		}

		newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
		for (int j = 0; j < 4; j++) {
			newData[nodeID + j + 1] = newNodeData[j];
		}

		// Reset quadtree data
		globalIndices_ = newGlobalIndices;
		parentIndices_ = newParentIndices;
		childIndices_ = newChildIndices;
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

		// Erase child nodes from level arrays
		int l = firstChildNode.level;
		int i = firstChildNode.levelID;
		globalIndices_[l].erase(globalIndices_[l].begin() + i, globalIndices_[l].begin() + i + 4);
		parentIndices_[l].erase(parentIndices_[l].begin() + i, parentIndices_[l].begin() + i + 4);
		childIndices_[l].erase(childIndices_[l].begin() + i, childIndices_[l].begin() + i + 4);

		// Iterate through tree via level arrays and create new ones
		LevelArray newGlobalIndices = globalIndices_;
		LevelArray newParentIndices = parentIndices_;
		LevelArray newChildIndices = childIndices_;
		std::vector<T> newData(data_.size() - 4);
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] > nodeID) {
					newGlobalIndices[l][i] = globalIndices_[l][i] - 4;
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
		data_ = newData;

		return;

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
		globalIndices[quadrant->level].push_back(wrapper->global_counter++);

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

        for (auto l = 0; l < nLevels; l++) {
            globalIndices_.push_back(std::vector<int>{});
            parentIndices_.push_back(std::vector<int>{});
            childIndices_.push_back(std::vector<int>{});

            globalIndices_[l].reserve((std::size_t) pow(2, 2*l));
            parentIndices_[l].reserve((std::size_t) pow(2, 2*l));
            childIndices_[l].reserve((std::size_t) pow(2, 2*l));
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

};

#endif

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_