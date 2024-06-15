#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>

#include <SpecialMatrices.hpp>
#include <EllipticForest.hpp>
#include <P4est.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

using namespace EllipticForest;

static double REFINE_FACTOR = 0.25;
static double COARSEN_FACTOR = 2.0;

class DoubleNodeFactory : public MPI::MPIObject, public AbstractNodeFactory<double> {

public:

    DoubleNodeFactory() :
        MPIObject(MPI_COMM_WORLD)
            {}
    
    DoubleNodeFactory(MPI::Communicator comm) :
        MPIObject(comm)
            {}

    virtual Node<double>* createNode(double data, std::string path, int level, int pfirst, int plast) {
        return new Node<double>(this->getComm(), data, path, level, pfirst, plast);
    }

    virtual Node<double>* createChildNode(Node<double>* parent_node, int sibling_id, int pfirst, int plast) {
        double child_data = REFINE_FACTOR*parent_node->data;
        std::string child_path = parent_node->path + std::to_string(sibling_id);
        int child_level = parent_node->level + 1;
        return new Node<double>(this->getComm(), child_data, child_path, child_level, pfirst, plast);
    }

    virtual Node<double>* createParentNode(std::vector<Node<double>*> children_nodes, int pfirst, int plast) {
        double parent_data = 0;
        for (auto* child : children_nodes) { parent_data += COARSEN_FACTOR*child->data; }
        std::string parent_path = children_nodes[0]->path.substr(0, children_nodes[0]->path.length()-1);
        int parent_level = children_nodes[0]->level - 1;
        return new Node<double>(this->getComm(), parent_data, parent_path, parent_level, pfirst, plast);
    }

};

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::Communicator comm = MPI_COMM_WORLD;
    MPI::MPIObject mpi(comm);
    // p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(0, 1, 0, 1);
    // p4est_t* p4est_obj = p4est_new_ext(comm, conn, 0, 0, true, 0, nullptr, nullptr);
    // p4est_refine(
    //     p4est_obj,
    //     1,
    //     [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //         std::cout << "HERE 1" << std::endl;
    //         return 1;
    //     },
    //     [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //         std::cout << "HERE 2" << std::endl;
    //         return;
    //     }
    // );

    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);
    int max_level = 2;
    app.log("Calling refine...");
    quadtree.refine(true,
        [&](Node<double>* node){
            // app.log("[refine] node-path: " + node->path + ", data: %f", node->data);
            return (int) node->level < max_level;
        }
    );

    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log("[traverse] node-path: " + node->path + ", data: %f", node->data);
            return 1;
        }
    );

    app.log("Calling partition...");
    quadtree.partition();

    // quadtree.traversePreOrder(
    //     [&](Node<double>* node){
    //         app.log("[traverse] node-path: " + node->path + ", data: %f", node->data);
    //         return 1;
    //     }
    // );

    return EXIT_SUCCESS;
}