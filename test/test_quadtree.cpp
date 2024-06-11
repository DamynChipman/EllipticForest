#include "gtest/gtest.h"
#include <EllipticForestApp.hpp>
#include <Quadtree.hpp>

using namespace EllipticForest;

const double REFINE_FACTOR = 0.25;
const double COARSEN_FACTOR = 1.0;

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

TEST(Quadtree, init) {

    MPI::Communicator comm = MPI_COMM_SELF;
    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);

    EXPECT_TRUE(quadtree.map["0"]->leaf);
    EXPECT_EQ(quadtree.map["0"]->data, root_data);
    EXPECT_EQ(quadtree.root(), root_data);

    std::vector<double> exp_data = {1000.0};

    int counter = 0;
    quadtree.traversePreOrder([&](Node<double>* node){
        EXPECT_EQ(exp_data[counter++], node->data);
        return 1;
    });

}

TEST(Quadtree, refine_coarsen_p4est_backend_recursive) {

    MPI::Communicator comm = MPI_COMM_SELF;
    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);

    int min_level, max_level, counter;

    max_level = 2;
    std::vector<double> exp_data = {
        root_data,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
    };

    quadtree.refine(true,
        [&](Node<double>* node){
            return (int) node->level < max_level;
        }
    );

    counter = 0;
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(exp_data[counter++], node->data);
            return 1;
        }
    );

    min_level = 0;
    quadtree.coarsen(true,
        [&](std::vector<Node<double>*> nodes){
            for (auto* node : nodes)
                return (int) node->level > min_level;
        }
    );

    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(root_data, node->data);
            return 1;
        }
    );

}

TEST(Quadtree, refine_coarsen_p4est_backend_individual) {

    MPI::Communicator comm = MPI_COMM_SELF;
    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);

    int counter;

    std::vector<double> exp_data = {
        root_data,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
    };

    quadtree.refineNode("0", true);
    quadtree.refineNode("00", true);
    quadtree.refineNode("01", true);
    quadtree.refineNode("02", true);
    quadtree.refineNode("03", true);

    counter = 0;
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(exp_data[counter++], node->data);
            return 1;
        }
    );

    quadtree.coarsenNode("02", true);
    quadtree.coarsenNode("01", true);
    quadtree.coarsenNode("03", true);
    quadtree.coarsenNode("00", true);
    quadtree.coarsenNode("0", true);

    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(root_data, node->data);
            return 1;
        }
    );

}

TEST(Quadtree, refine_coarsen_external_backend) {

    MPI::Communicator comm = MPI_COMM_SELF;
    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);

    int counter;

    std::vector<double> exp_data = {
        root_data,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
        root_data*REFINE_FACTOR*REFINE_FACTOR,
    };

    p4est_refine(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return (int) (EllipticForest::p4est::p4est_quadrant_path(quadrant) == "0");
        },
        nullptr
    );
    quadtree.refineNode("0", false);

    p4est_refine(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return (int) (EllipticForest::p4est::p4est_quadrant_path(quadrant) == "00");
        },
        nullptr
    );
    quadtree.refineNode("00", false);
    
    p4est_refine(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return (int) (EllipticForest::p4est::p4est_quadrant_path(quadrant) == "01");
        },
        nullptr
    );
    quadtree.refineNode("01", false);

    p4est_refine(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return (int) (EllipticForest::p4est::p4est_quadrant_path(quadrant) == "02");
        },
        nullptr
    );
    quadtree.refineNode("02", false);

    p4est_refine(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
            return (int) (EllipticForest::p4est::p4est_quadrant_path(quadrant) == "03");
        },
        nullptr
    );
    quadtree.refineNode("03", false);

    counter = 0;
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(exp_data[counter++], node->data);
            return 1;
        }
    );

    p4est_coarsen(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]){
            std::string first_child_path = EllipticForest::p4est::p4est_quadrant_path(quadrants[0]);
            std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
            return (int) (parent_path == "02");
        },
        nullptr
    );
    quadtree.coarsenNode("02");

    p4est_coarsen(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]){
            std::string first_child_path = EllipticForest::p4est::p4est_quadrant_path(quadrants[0]);
            std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
            return (int) (parent_path == "03");
        },
        nullptr
    );
    quadtree.coarsenNode("03");

    p4est_coarsen(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]){
            std::string first_child_path = EllipticForest::p4est::p4est_quadrant_path(quadrants[0]);
            std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
            return (int) (parent_path == "01");
        },
        nullptr
    );
    quadtree.coarsenNode("01");

    p4est_coarsen(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]){
            std::string first_child_path = EllipticForest::p4est::p4est_quadrant_path(quadrants[0]);
            std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
            return (int) (parent_path == "00");
        },
        nullptr
    );
    quadtree.coarsenNode("00");

    p4est_coarsen(
        quadtree.p4est,
        0,
        [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrants[]){
            std::string first_child_path = EllipticForest::p4est::p4est_quadrant_path(quadrants[0]);
            std::string parent_path = first_child_path.substr(0, first_child_path.length() - 1);
            return (int) (parent_path == "0");
        },
        nullptr
    );
    quadtree.coarsenNode("0");

    quadtree.traversePreOrder(
        [&](Node<double>* node){
            EXPECT_EQ(root_data, node->data);
            return 1;
        }
    );

}

TEST(Quadtree, propagate) {
    MPI::Communicator comm = MPI_COMM_SELF;
    DoubleNodeFactory node_factory(comm);
    double root_data = 1000.0;
    Quadtree<double> quadtree(comm, root_data, node_factory);

    int min_level, max_level, counter;

    max_level = 3;

    quadtree.refine(true,
        [&](Node<double>* node){
            return (int) node->level < max_level;
        }
    );

    int n_calls = 0;
    quadtree.propagate(
        "0132",
        [&](Node<double>* parent_node, std::vector<Node<double>*> children_nodes){
            std::cout << "parent: " << parent_node->path << std::endl;
            for (auto* child_node : children_nodes) {
                std::cout << "  child: " << child_node->path << std::endl;
            }
            std::cout << std::endl;
            n_calls++; 
            return 1;
        }
    );
    EXPECT_EQ(n_calls, max_level);
}