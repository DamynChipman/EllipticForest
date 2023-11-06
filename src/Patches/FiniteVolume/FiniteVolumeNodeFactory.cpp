#include "FiniteVolumeNodeFactory.hpp"

namespace EllipticForest {

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory() {}

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory(MPI_Comm comm) :
    MPIObject(comm)
        {}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createNode(FiniteVolumePatch data, std::string path, int level, int pfirst, int plast) {
    return new Node<FiniteVolumePatch>(this->getComm(), data, path, level, pfirst, plast);
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createChildNode(Node<FiniteVolumePatch>* parentNode, int siblingID, int pfirst, int plast) {
    
    // auto& app = EllipticForest::EllipticForestApp::getInstance();
    // app.log("In creatChildNode: parent_node = %s", parentNode->path.c_str());
    // Get parent grid info
    auto& parentGrid = parentNode->data.grid();
    int nx = parentGrid.nx();
    int ny = parentGrid.ny();
    double xLower = parentGrid.xLower();
    double xUpper = parentGrid.xUpper();
    double xMid = (xLower + xUpper) / 2.0;
    double yLower = parentGrid.yLower();
    double yUpper = parentGrid.yUpper();
    double yMid = (yLower + yUpper) / 2.0;

    // Create grid communicator
    // app.log("Creating sub-communicator...");
    // MPI::Communicator child_comm;
    // MPI::communicatorSubsetRange(parentNode->getComm(), pfirst, plast, 0, child_comm);
    // app.log("Done.");

    // Create child grid
    FiniteVolumeGrid child_grid;
    switch (siblingID) {
        case 0:
            // Lower left
            child_grid = FiniteVolumeGrid(parentNode->data.getComm(), nx, xLower, xMid, ny, yLower, yMid);
            break;
        case 1:
            // Lower right
            child_grid = FiniteVolumeGrid(parentNode->data.getComm(), nx, xMid, xUpper, ny, yLower, yMid);
            break;
        case 2:
            // Upper left
            child_grid = FiniteVolumeGrid(parentNode->data.getComm(), nx, xLower, xMid, ny, yMid, yUpper);
            break;
        case 3:
            // Upper right
            child_grid = FiniteVolumeGrid(parentNode->data.getComm(), nx, xMid, xUpper, ny, yMid, yUpper);
            break;
        default:
            break;
    }

    // Create child patch
    FiniteVolumePatch childPatch(parentNode->data.getComm(), child_grid);

    // Create child node
    std::string path = parentNode->path + std::to_string(siblingID);
    int level = parentNode->level + 1;
    return new Node<FiniteVolumePatch>(this->getComm(), childPatch, path, level, pfirst, plast);
    
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createParentNode(std::vector<Node<FiniteVolumePatch>*> childNodes, int pfirst, int plast) {

    // Create parent grid
    int nx = childNodes[0]->data.grid().nx();
    int ny = childNodes[0]->data.grid().ny();
    double xLower = childNodes[0]->data.grid().xLower();
    double xUpper = childNodes[1]->data.grid().xUpper();
    double yLower = childNodes[0]->data.grid().yLower();
    double yUpper = childNodes[2]->data.grid().yUpper();
    FiniteVolumeGrid parentGrid(this->getComm(), nx, xLower, xUpper, ny, yLower, yUpper);

    // Create communicator for parent patch
    // MPI::Group alpha_beta_group;
    // MPI::Group gamma_omega_group;
    

    // Create parent patch
    FiniteVolumePatch parentPatch(this->getComm(), parentGrid); // TODO: Switch MPI_COMM_WORLD to patch communicator

    // Create parent node
    std::string path = childNodes[0]->path.substr(0, childNodes[0]->path.length()-1);
    int level = childNodes[0]->level - 1;
    return new Node<FiniteVolumePatch>(this->getComm(), parentPatch, path, level, pfirst, plast);

}

} // NAMESPACE : EllipticForest