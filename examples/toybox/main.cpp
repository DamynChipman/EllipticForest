#include <iostream>
#include <string>
#include <vector>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

using namespace EllipticForest;

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
        double child_data = parent_node->data / 4.0;
        std::string child_path = parent_node->path + std::to_string(sibling_id);
        int child_level = parent_node->level + 1;
        return new Node<double>(this->getComm(), child_data, child_path, child_level, pfirst, plast);
    }

    virtual Node<double>* createParentNode(std::vector<Node<double>*> children_nodes, int pfirst, int plast) {
        double parent_data = 0;
        for (auto* child : children_nodes) { parent_data += 10.0*child->data; }
        std::string parent_path = children_nodes[0]->path.substr(0, children_nodes[0]->path.length()-1);
        int parent_level = children_nodes[0]->level - 1;
        return new Node<double>(this->getComm(), parent_data, parent_path, parent_level, pfirst, plast);
    }

};

int main(int argc, char** argv) {

    EllipticForestApp app{&argc, &argv};
    MPI::MPIObject mpi{};

    int min_level = 1;
    int max_level = 2;
    double root_data = 100.0;
    DoubleNodeFactory node_factory(mpi.getComm());

    Quadtree<double> quadtree(mpi.getComm(), root_data, node_factory);

    app.log("Initial quadtree:");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.refineNode("0", true);
    app.log("Refined '0'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.refineNode("00", true);
    app.log("Refined '00'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.refineNode("03", true);
    app.log("Refined '03'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.coarsenNode("03", true);
    app.log("Coarsened '03'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.coarsenNode("00", true);
    app.log("Coarsened '00'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.coarsenNode("0", true);
    app.log("Coarsened '0'");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

#if 0
    quadtree.refine(true,
        [&](Node<double>* node){
            return (int) node->level < max_level;
        }
    );
    quadtree.partition();

    app.log("Refined to level " + std::to_string(max_level) + ":");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    quadtree.coarsen(true,
        [&](std::vector<Node<double>*> nodes){
            for (auto* node : nodes)
                return (int) node->level > min_level;
        }
    );

    app.log("Coarsened to level " + std::to_string(min_level) + ":");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );

    max_level = 4;
    quadtree.refine(true,
        [&](Node<double>* node){
            return (int) node->level < max_level;
        }
    );

    app.log("Refined to level " + std::to_string(max_level) + ":");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
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

    app.log("Coarsened to level " + std::to_string(min_level) + ":");
    quadtree.traversePreOrder(
        [&](Node<double>* node){
            app.log(node->path + " : " + std::to_string(node->data));
            return 1;
        }
    );
#endif
    return EXIT_SUCCESS;
}