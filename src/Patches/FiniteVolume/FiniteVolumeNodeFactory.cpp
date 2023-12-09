#include "FiniteVolumeNodeFactory.hpp"

namespace EllipticForest {

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory() {}

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory(MPI_Comm comm) :
    MPIObject(comm)
        {}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createNode(FiniteVolumePatch data, std::string path, int level, int pfirst, int plast) {
    return new Node<FiniteVolumePatch>(this->getComm(), data, path, level, pfirst, plast);
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createChildNode(Node<FiniteVolumePatch>* parent_node, int sibling_id, int pfirst, int plast) {
    
    // auto& app = EllipticForest::EllipticForestApp::getInstance();
    // app.log("In creatChildNode: parent_node = %s", parent_node->path.c_str());
    // Get parent grid info
    auto& parent_grid = parent_node->data.grid();
    int nx = parent_grid.nx();
    int ny = parent_grid.ny();
    double x_lower = parent_grid.xLower();
    double x_upper = parent_grid.xUpper();
    double x_mid = (x_lower + x_upper) / 2.0;
    double y_lower = parent_grid.yLower();
    double y_upper = parent_grid.yUpper();
    double y_mid = (y_lower + y_upper) / 2.0;

    // Create grid communicator
    // app.log("Creating sub-communicator...");
    // MPI::Communicator child_comm;
    // MPI::communicatorSubsetRange(parent_node->getComm(), pfirst, plast, 0, child_comm);
    // app.log("Done.");

    // Create child grid
    FiniteVolumeGrid child_grid;
    switch (sibling_id) {
        case 0:
            // Lower left
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_lower, x_mid, ny, y_lower, y_mid);
            break;
        case 1:
            // Lower right
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_mid, x_upper, ny, y_lower, y_mid);
            break;
        case 2:
            // Upper left
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_lower, x_mid, ny, y_mid, y_upper);
            break;
        case 3:
            // Upper right
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_mid, x_upper, ny, y_mid, y_upper);
            break;
        default:
            break;
    }

    // Create child patch
    FiniteVolumePatch child_patch(parent_node->data.getComm(), child_grid);

    // Create child node
    std::string path = parent_node->path + std::to_string(sibling_id);
    int level = parent_node->level + 1;
    return new Node<FiniteVolumePatch>(this->getComm(), child_patch, path, level, pfirst, plast);
    
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createParentNode(std::vector<Node<FiniteVolumePatch>*> child_nodes, int pfirst, int plast) {

    // Create parent grid
    int nx = child_nodes[0]->data.grid().nx();
    int ny = child_nodes[0]->data.grid().ny();
    double x_lower = child_nodes[0]->data.grid().xLower();
    double x_upper = child_nodes[1]->data.grid().xUpper();
    double y_lower = child_nodes[0]->data.grid().yLower();
    double y_upper = child_nodes[2]->data.grid().yUpper();
    FiniteVolumeGrid parent_grid(this->getComm(), nx, x_lower, x_upper, ny, y_lower, y_upper);

    // Create communicator for parent patch
    // MPI::Group alpha_beta_group;
    // MPI::Group gamma_omega_group;
    

    // Create parent patch
    FiniteVolumePatch parentPatch(this->getComm(), parent_grid); // TODO: Switch MPI_COMM_WORLD to patch communicator

    // Create parent node
    std::string path = child_nodes[0]->path.substr(0, child_nodes[0]->path.length()-1);
    int level = child_nodes[0]->level - 1;
    return new Node<FiniteVolumePatch>(this->getComm(), parentPatch, path, level, pfirst, plast);

}

} // NAMESPACE : EllipticForest