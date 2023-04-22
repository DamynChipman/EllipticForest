#ifndef QUADTREE_HPP_
#define QUADTREE_HPP_

#include <iostream>
#include <string>
#include <ostream>
#include <vector>
#include <functional>
#include <p4est.h>
#include <p4est_search.h>

namespace EllipticForest {

template<typename T>
class Quadtree {

public:

    typedef std::vector<std::vector<int>> LevelArray;

protected:

	p4est_t* p4est_;
    LevelArray globalIndices_{};
    LevelArray parentIndices_{};
    LevelArray childIndices_{};
    std::vector<T> data_{};
    // std::function<T(std::size_t, std::size_t)> initializationFunction_;

    typedef struct quadtree_data_wrapper {

		LevelArray* g;
		LevelArray* p;
		LevelArray* c;
		std::vector<T>* d;
		int global_counter;

	} quadtree_data_wrapper_t;

    typedef struct quadtree_traversal_wrapper {

		std::function<void(T&)> visit;
		T& data;

	} quadtree_traversal_wrapper_t;

	struct QuadtreeNode {
		T data;
		int level;
		int globalID;
		int parentID;
		std::vector<int> childrenIDs{4};
	};

public:

    Quadtree() :
		p4est_(nullptr)
			{}
	
	Quadtree(p4est_t* p4est) :
		p4est_(p4est) {

		// buildLevelArrays_();
		// buildData_();
	
	}

	LevelArray globalIndices() const { return globalIndices_; }
	LevelArray parentIndices() const { return parentIndices_; }
	LevelArray childIndices() const { return childIndices_; }
	T& root() { return data_[0]; }
	const T& root() const { return data_[0]; }
	std::vector<T>& data() { return data_; }

	QuadtreeNode getNode(int ID) {

		QuadtreeNode node;
		int i;

		// Find node in global level array
		bool breakLoop = false;
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] == ID) {
					// Node found, get data and return
					node.level = l;
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
	 * @brief User derived function for data initialization given the node's level and index
	 * 
	 * @param parentData Node's parent data
	 * @param level Node's level in tree
	 * @param index Node's index in level
	 * @return T A newly constructed node data
	 */
	// virtual T initData(T& parentData, std::size_t level, std::size_t index) = 0;
	// virtual void toVTK(std::string filename) = 0;

	void build(T rootData, std::function<T(T& parentNode, std::size_t childIndex)> initDataFunction) {
		buildLevelArrays_();
		buildData_(rootData, initDataFunction);
	}

	void buildFromP4est(p4est_t* p4est, T rootData, std::function<T(T& parentNode, int childIndex)> initDataFunction) {
		p4est_ = p4est;
		buildLevelArrays_();
		buildData_(rootData, initDataFunction);
	}

	void buildFromRoot(T rootData) {
		data_.push_back(rootData);
		globalIndices_.push_back({0});
		parentIndices_.push_back({-1});
		childIndices_.push_back({-1});
	}

	void traversePreOrder(std::function<void(T&)> visit) {

		for (auto& d : data_) visit(d);

	}

	void traversePostOrder(std::function<void(T&)> visit) {

		_traversePostOrder(visit, 0, 0);

	}

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

	void refineNode(int nodeID, std::function<std::vector<T>(T&)> parent2childrenFunction) {

		// Get node information
		QuadtreeNode node = getNode(nodeID);

		if (node.childrenIDs[0] != -1) {
			return;
		}

		// Create new fine nodes (children)
		T& p = data_[node.globalID];
		std::vector<T> newNodeData = parent2childrenFunction(p);

		// Iterate through tree via level arrays and create new ones
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
					std::cout << "Old globalID = " << globalIndices_[l][i] << std::endl;
					std::cout << "New globalID = " << newGlobalIndices[l][i] << std::endl;
					newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
				}

				// if (globalIndices_[l][i] > nodeID) {
				// 	// Unaffected node after node to be refined;

				// 	if (l == node.level+1) {
				// 		newGlobalIndices[l][i+shift] = globalIndices_[l][i] + 4;
				// 		newParentIndices[l][i+shift] = parentIndices_[l][i] + 4;
				// 		// newChildIndices[l][i+shift] = childIndices_[l][i] + 4;
				// 		newData[newGlobalIndices[l][i]] = std::move(data_[globalIndices_[l][i]]);
				// 	}
				// 	else {
				// 		newGlobalIndices[l][i] = globalIndices_[l][i] + 4;
				// 		// newChildIndices[l][i] = childIndices_[l][i] + 4;
				// 		// newParentIndices[l][i] = parentIndices_[l][i] + 4;
				// 		newData[newGlobalIndices[l][i]] = std::move(data_[globalIndices_[l][i]]);
				// 	}

				// 	if (childIndices_[l][i] != -1) {
				// 		newChildIndices[l][i] = childIndices_[l][i] + 4;
				// 	}

				// 	// newParentIndices[l][i] = parentIndices_[l][i] + 4;

				// }
				// else if (globalIndices_[l][i] == nodeID) {
				// 	// Node to be refined;

				// 	newChildIndices[l][i] = nodeID + 1;
				// 	newData[newGlobalIndices[l][i]] = std::move(data_[globalIndices_[l][i]]);

					

				// 	if (l == globalIndices_.size()-1) {
				// 		// New level needs to be created
				// 		std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
				// 		std::vector<int> childrenPIDS(4, nodeID);
				// 		std::vector<int> childrenCIDS(4, -1);

				// 		newGlobalIndices.push_back(childrenGIDS);
				// 		newParentIndices.push_back(childrenPIDS);
				// 		newChildIndices.push_back(childrenCIDS);
				// 	}
				// 	else {
				// 		// Nodes inserted before ones on same level
				// 		int nextLevelFirstChildID = -1;
				// 		int j = 1;
				// 		while (nextLevelFirstChildID == -1) {
				// 			nextLevelFirstChildID = newChildIndices[l][i+j];
				// 			j++;
				// 		}
				// 		QuadtreeNode nextLevelFirstChildNode = getNode(nextLevelFirstChildID);

				// 		std::vector<int>::iterator iterG = std::find(newGlobalIndices[l+1].begin(), newGlobalIndices[l+1].end(), nextLevelFirstChildNode.globalID);
				// 		std::vector<int>::iterator iterP = std::find(newParentIndices[l+1].begin(), newParentIndices[l+1].end(), nextLevelFirstChildNode.parentID);
				// 		std::vector<int>::iterator iterC = std::find(newChildIndices[l+1].begin(), newChildIndices[l+1].end(), nextLevelFirstChildNode.childrenIDs[0]);

				// 		std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
				// 		std::vector<int> childrenPIDS(4, nodeID);
				// 		std::vector<int> childrenCIDS(4, -1);
						
				// 		newGlobalIndices[l+1].insert(iterG, childrenGIDS.begin(), childrenGIDS.end());
				// 		newParentIndices[l+1].insert(iterP, childrenPIDS.begin(), childrenPIDS.end());
				// 		newChildIndices[l+1].insert(iterC, childrenCIDS.begin(), childrenCIDS.end());

				// 		shift = 4;
				// 	}
					
				// 	for (int j = 0; j < 4; j++) {
				// 		newData[newGlobalIndices[l+1][j]] = std::move(newNodeData[j]);
				// 	}

				// }
				// else {
				// 	// Unaffected node before node to be refined;

				// 	newData[newGlobalIndices[l][i]] = std::move(data_[globalIndices_[l][i]]);
				// }

				// std::cout << "===================================" << std::endl;
				// std::cout << "L = " << l << ", i = " << i << std::endl;
				// std::cout << "Global Indices:" << std::endl;
				// for (auto ll : newGlobalIndices) {
				// 	for (auto ii : ll) {
				// 		std::cout << ii << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				// std::cout << "Parent Indices:" << std::endl;
				// for (auto ll : newParentIndices) {
				// 	for (auto ii : ll) {
				// 		std::cout << ii << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				// std::cout << "Child Indices:" << std::endl;
				// for (auto ll : newChildIndices) {
				// 	for (auto ii : ll) {
				// 		std::cout << ii << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				// std::cout << "===================================" << std::endl;
			}
		}

		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] == nodeID) {

					newChildIndices[l][i] = nodeID + 1;

					if (l == globalIndices_.size()-1) {
						// New level needs to be created
						std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
						std::vector<int> childrenPIDS(4, nodeID);
						std::vector<int> childrenCIDS(4, -1);

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

						std::vector<int> childrenGIDS = {nodeID + 1, nodeID + 2, nodeID + 3, nodeID + 4};
						std::vector<int> childrenPIDS(4, nodeID);
						std::vector<int> childrenCIDS(4, -1);
						
						newGlobalIndices[l+1].insert(iterG, childrenGIDS.begin(), childrenGIDS.end());
						newParentIndices[l+1].insert(iterP, childrenPIDS.begin(), childrenPIDS.end());
						newChildIndices[l+1].insert(iterC, childrenCIDS.begin(), childrenCIDS.end());
					}

					printf("l = %i, i = %i, gOld = %i, gNew = %i\n", l, i, globalIndices_[l][i], newGlobalIndices[l][i]);
					newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
					for (int j = 0; j < 4; j++) {
						newData[nodeID + j + 1] = newNodeData[j];
					}

					// newData[newGlobalIndices[l][i]] = data_[globalIndices_[l][i]];
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

	void coarsenNode(int nodeID, std::function<T(T&, T&, T&, T&)> children2parentFunction) {

		// Get node information
		QuadtreeNode node = getNode(nodeID);

		// Create new coarse node (parent)
		T& c0 = data_[node.childrenIDs[0]];
		T& c1 = data_[node.childrenIDs[1]];
		T& c2 = data_[node.childrenIDs[2]];
		T& c3 = data_[node.childrenIDs[3]];
		T newNodeData = children2parentFunction(c0, c1, c2, c3);

		// Iterate through tree via level arrays and create new ones
		LevelArray newGlobalIndices = globalIndices_;
		LevelArray newParentIndices = parentIndices_;
		LevelArray newChildIndices = childIndices_;
		std::vector<T> newData(data_.size() - 4);
		for (int l = 0; l < globalIndices_.size(); l++) {
			for (int i = 0; i < globalIndices_[l].size(); i++) {
				if (globalIndices_[l][i] == node.childrenIDs[0] || globalIndices_[l][i] == node.childrenIDs[1] || globalIndices_[l][i] == node.childrenIDs[2] || globalIndices_[l][i] == node.childrenIDs[3]) {
					// Child node; erase from all level arrays

					newGlobalIndices[l].erase(newGlobalIndices[l].begin() + i);
					newParentIndices[l].erase(newParentIndices[l].begin() + i);
					newChildIndices[l].erase(newChildIndices[l].begin() + i);

				}
				else if (globalIndices_[l][i] > nodeID) {
					// Unaffected node after node to be coarsened; copy old node data to new data array after node to be coarsened

					newGlobalIndices[l][i] = newGlobalIndices[l][i] - 4;
					newData[newGlobalIndices[l][i]] = std::move(data_[newGlobalIndices[l][i] + 4]);

				}
				else if (globalIndices_[l][i] == nodeID) {
					// Node to be coarsened; remove child index and place new node data where old node data was

					newChildIndices[l][i] = -1;
					newData[newGlobalIndices[l][i]] = newNodeData;

				}
				else {
					// Unaffected node before node to be coarsened; copy old node data to new data array at same location

					newData[newGlobalIndices[l][i]] = std::move(data_[newGlobalIndices[l][i]]);

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

    static int p4est_visit_pre(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {

		// Get access to level arrays
		quadtree_data_wrapper_t* wrapper = (quadtree_data_wrapper_t*) p4est->user_pointer;
		LevelArray& globalIndices = *(wrapper->g);
		LevelArray& parentIndices = *(wrapper->p);
		LevelArray& childIndices = *(wrapper->c);
		std::vector<T> data = *(wrapper->d);

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

	static int p4est_visit_post(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point) {
		
		// Get access to level arrays
		quadtree_data_wrapper_t* wrapper = (quadtree_data_wrapper_t*) p4est->user_pointer;
		LevelArray& globalIndices = *(wrapper->g);
		LevelArray& parentIndices = *(wrapper->p);
		LevelArray& childIndices = *(wrapper->c);
		std::vector<T> data = *(wrapper->d);

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
		p4est_->user_pointer = &wrapper;

        // Call `p4est_search_reorder` to traverse tree and populate level arrays
		int skipLevels = 0;
		p4est_search_reorder(p4est_, skipLevels, NULL, this->p4est_visit_pre, this->p4est_visit_post, NULL, NULL);

		// Restore original p4est user pointer
		p4est_->user_pointer = userPointerSaved;

    }

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

} // NAMESPACE : EllipticForest

#endif // QUADTREE_HPP_