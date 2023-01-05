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

#define ELLIPTIC_FOREST_QUADTREE_NUMBER_CHILDREN    4
#define ELLIPTIC_FOREST_QUADTREE_LOWER_LEFT 	    0
#define ELLIPTIC_FOREST_QUADTREE_LOWER_RIGHT 	    1
#define ELLIPTIC_FOREST_QUADTREE_UPPER_LEFT 	    2
#define ELLIPTIC_FOREST_QUADTREE_UPPER_RIGHT 	    3
#define ELLIPTIC_FOREST_QUADTREE_MAX_HEIGHT         40 // Arbitrary right now, change if needed

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

	/**
	 * @brief User derived function for data initialization given the node's level and index
	 * 
	 * @param parentData Node's parent data
	 * @param level Node's level in tree
	 * @param index Node's index in level
	 * @return T A newly constructed node data
	 */
	virtual T initData(T& parentData, std::size_t level, std::size_t index) = 0;
	virtual void toVTK(std::string filename) = 0;

	void build(T rootData) {
		buildLevelArrays_();
		buildData_(rootData);
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

	friend std::ostream& operator<<(std::ostream& os, const Quadtree& quadtree) {

		os << "Global Indices:" << std::endl;
		for (auto& g : quadtree.globalIndices()) {
			for (auto& i : g) {
				os << i << ",  ";
			}
			os << std::endl;
		}
		os << std::endl;

		os << "Parent Indices:" << std::endl;
		for (auto& p : quadtree.parentIndices()) {
			for (auto& i : p) {
				os << i << ",  ";
			}
			os << std::endl;
		}
		os << std::endl;

		os << "Child Indices:" << std::endl;
		for (auto& c : quadtree.childIndices()) {
			for (auto& i : c) {
				os << i << ",  ";
			}
			os << std::endl;
		}
		os << std::endl;

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

    void buildData_(T& rootData) {

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
					data_[globalIndices_[l][i]] = initData(parentData, l, i);
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