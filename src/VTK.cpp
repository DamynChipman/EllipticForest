#include "VTK.hpp"

namespace EllipticForest {

XMLNode* DataArrayNodeBase::toVTK() {
    XMLNode* nodeDataArray = new XMLNode("DataArray");
    nodeDataArray->addAttribute("type", getType());
    nodeDataArray->addAttribute("Name", getName());
    nodeDataArray->addAttribute("NumberOfComponents", getNumberOfComponents());
    nodeDataArray->addAttribute("format", getFormat());
    nodeDataArray->addAttribute("RangeMin", getRangeMin());
    nodeDataArray->addAttribute("RangeMax", getRangeMax());
    nodeDataArray->data = getData();
    return nodeDataArray;
}

XMLNode* DataArrayNodeBase::toPVTK() {
    XMLNode* nodeDataArray = new XMLNode("PDataArray");
    nodeDataArray->addAttribute("type", getType());
    nodeDataArray->addAttribute("Name", getName());
    nodeDataArray->addAttribute("NumberOfComponents", getNumberOfComponents());
    nodeDataArray->addAttribute("format", getFormat());
    nodeDataArray->data = "";
    return nodeDataArray;
}

EmptyDataArrayNode::EmptyDataArrayNode() {}
std::string EmptyDataArrayNode::getType() { return "Float32"; }
std::string EmptyDataArrayNode::getName() { return "Empty"; }
std::string EmptyDataArrayNode::getNumberOfComponents() { return "1"; }
std::string EmptyDataArrayNode::getFormat() { return "ascii"; }
std::string EmptyDataArrayNode::getRangeMin() { return "0.0"; }
std::string EmptyDataArrayNode::getRangeMax() { return "0.0"; }
std::string EmptyDataArrayNode::getData() { return "0.0"; }

RectilinearGridVTK::RectilinearGridVTK() : meshComplete_(false), coordsDataVector_(0), pointDataVector_(0), cellDataVector_(0), root_("VTKFile") {
    root_.addAttribute("type", "RectilinearGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

void RectilinearGridVTK::buildMesh(RectilinearGridNodeBase& mesh, DataArrayNodeBase* xCoords, DataArrayNodeBase* yCoords, DataArrayNodeBase* zCoords) {
    mesh_ = &mesh;
    
    if (xCoords != nullptr) {
        coordsDataVector_.push_back(xCoords);
    }
    else {
        coordsDataVector_.push_back(&emptyDataArray_);
    }

    if (yCoords != nullptr) {
        coordsDataVector_.push_back(yCoords);
    }
    else {
        coordsDataVector_.push_back(&emptyDataArray_);
    }

    if (zCoords != nullptr) {
        coordsDataVector_.push_back(zCoords);
    }
    else {
        coordsDataVector_.push_back(&emptyDataArray_);
    }

    meshComplete_ = true;
    return;
}

void RectilinearGridVTK::addPointData(DataArrayNodeBase& pointData) {
    pointDataVector_.push_back(&pointData);
}

void RectilinearGridVTK::addCellData(DataArrayNodeBase& cellData) {
    cellDataVector_.push_back(&cellData);
}

void RectilinearGridVTK::toVTK(std::string filename) {

    if (meshComplete_) {

        XMLNode nodeRectilinearGrid("RectilinearGrid");
        nodeRectilinearGrid.addAttribute("WholeExtent", mesh_->getWholeExtent());

        XMLNode nodePiece("Piece");
        nodePiece.addAttribute("Extent", mesh_->getExtent());

        XMLNode nodeCellData("CellData");
        if (!cellDataVector_.empty()) {
            std::string names = "";
            for (auto& cellData : cellDataVector_) names += cellData->getName() + " ";
            nodeCellData.addAttribute("Scalars", names);
        }
        // for (auto cellData : cellDataVector_) {
        //     nodeCellData.addAttribute("Scalars", cellData->getName());
        // }

        XMLNode nodePointData("PointData");
        if (!pointDataVector_.empty()) {
            std::string names = "";
            for (auto& pointData : pointDataVector_) names += pointData->getName() + " ";
            nodePointData.addAttribute("Scalars", names);
        }
        // for (auto pointData : pointDataVector_) {
        //     nodePointData.addAttribute("Scalars", pointData->getName());
        // }

        XMLNode nodeCoordinates("Coordinates");
        
        root_.addChild(nodeRectilinearGrid);
        nodeRectilinearGrid.addChild(nodePiece);
        if (!cellDataVector_.empty()) nodePiece.addChild(nodeCellData);
        if (!pointDataVector_.empty()) nodePiece.addChild(nodePointData);
        nodePiece.addChild(nodeCoordinates);
        for (auto cellData : cellDataVector_) nodeCellData.addChild(cellData->toVTK());
        for (auto pointData : pointDataVector_) nodePointData.addChild(pointData->toVTK());
        for (auto coordData : coordsDataVector_) nodeCoordinates.addChild(coordData->toVTK());
        // for (auto i = 0; i < coordsDataVector_.size(); i++) nodeCoordinates.addChild(coordsDataVector_[i]->toVTK());

        XMLTree tree(root_);
        tree.write(filename);

    }
    else {
        throw std::invalid_argument("[RectilinearGridVTK::toVTK] mesh and data not complete.");
    }

}

UnstructuredGridVTK::UnstructuredGridVTK() :
    meshComplete_(false),
    pointDataVector_(0),
    cellDataVector_(0),
    root_("VTKFile") {
    root_.addAttribute("type", "UnstructuredGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

UnstructuredGridNodeBase& UnstructuredGridVTK::mesh() {
    return *mesh_;
}

std::vector<DataArrayNodeBase*>& UnstructuredGridVTK::pointDataVector() {
    return pointDataVector_;
}

std::vector<DataArrayNodeBase*>& UnstructuredGridVTK::cellDataVector() {
    return cellDataVector_;
}

void UnstructuredGridVTK::buildMesh(UnstructuredGridNodeBase& mesh) {

    mesh_ = &mesh;
    meshComplete_ = true;

}

void UnstructuredGridVTK::addPointData(DataArrayNodeBase& pointData) {
    pointDataVector_.push_back(&pointData);
}

void UnstructuredGridVTK::addCellData(DataArrayNodeBase& cellData) {
    cellDataVector_.push_back(&cellData);
}

void UnstructuredGridVTK::toVTK(std::string filename) {

    if (meshComplete_) {

        XMLNode nodeUnstructuredGrid("UnstructuredGrid");

        XMLNode nodePiece("Piece");
        nodePiece.addAttribute("NumberOfPoints", mesh_->getNumberOfPoints());
        nodePiece.addAttribute("NumberOfCells", mesh_->getNumberOfCells());

        XMLNode nodePoints("Points");
        XMLNode nodeCells("Cells");

        XMLNode nodeCellData("CellData");
        if (!cellDataVector_.empty()) {
            std::string names = "";
            for (auto& cellData : cellDataVector_) names += cellData->getName() + " ";
            nodeCellData.addAttribute("Scalars", names);
        }

        XMLNode nodePointData("PointData");
        if (!pointDataVector_.empty()) {
            std::string names = "";
            for (auto& pointData : pointDataVector_) names += pointData->getName() + " ";
            nodePointData.addAttribute("Scalars", names);
        }

        root_.addChild(nodeUnstructuredGrid);
            nodeUnstructuredGrid.addChild(nodePiece);
                nodePiece.addChild(nodePoints);
                    nodePoints.addChild(mesh_->getPoints().toVTK());
                nodePiece.addChild(nodeCells);
                    nodeCells.addChild(mesh_->getConnectivity().toVTK());
                    nodeCells.addChild(mesh_->getOffsets().toVTK());
                    nodeCells.addChild(mesh_->getTypes().toVTK());
                if (!cellDataVector_.empty()) nodePiece.addChild(nodeCellData);
                    for (auto& cellData : cellDataVector_) nodeCellData.addChild(cellData->toVTK());
                if (!pointDataVector_.empty()) nodePiece.addChild(nodePointData);
                    for (auto& pointData : pointDataVector_) nodePointData.addChild(pointData->toVTK());

        XMLTree tree(root_);
        tree.write(filename);
        
    }
    else {
        throw std::invalid_argument("[UnstructuredGridVTK::toVTK] mesh and data not complete");
    }

}

PUnstructuredGridVTK::PUnstructuredGridVTK() :
    MPIObject{MPI_COMM_WORLD},
    root_("VTKFile") {
    root_.addAttribute("type", "UnstructuredGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

PUnstructuredGridVTK::PUnstructuredGridVTK(MPI_Comm comm) :
    MPIObject{comm},
    root_("VTKFile") {
    root_.addAttribute("type", "UnstructuredGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

void PUnstructuredGridVTK::buildMesh(UnstructuredGridNodeBase& mesh) {
    vtu.buildMesh(mesh);
}

void PUnstructuredGridVTK::addPointData(DataArrayNodeBase& pointData) {
    vtu.addPointData(pointData);
}

void PUnstructuredGridVTK::addCellData(DataArrayNodeBase& cellData) {
    vtu.addCellData(cellData);
}

void PUnstructuredGridVTK::toVTK(std::string filenameBase) {

    // All ranks write individual file
    char str_buffer[256];
    std::string filename = filenameBase + "_%04i.vtu";
    snprintf(str_buffer, 256, filename.c_str(), this->getRank());
    filename = std::string(str_buffer);
    vtu.toVTK(filename);

    // Head rank writes metadata file
    if (this->getRank() == MPI::HEAD_RANK) {

        // Get references to mesh and data
        auto& mesh = vtu.mesh();
        auto& pointDataVector = vtu.pointDataVector();
        auto& cellDataVector = vtu.cellDataVector();

        // Build nodes
        XMLNode nodePUnstructuredGrid("PUnstructuredGrid");
        nodePUnstructuredGrid.addAttribute("GhostLevel", "0");

        XMLNode nodePPoints("PPoints");
        
        XMLNode nodePPointData("PPointData");
        if (!pointDataVector.empty()) {
            std::string names = "";
            for (auto& pointData : pointDataVector) names += pointData->getName() + " ";
            nodePPointData.addAttribute("Scalars", names);
        }

        XMLNode nodePCellData("PCellData");
        if (!cellDataVector.empty()) {
            std::string names = "";
            for (auto& cellData : cellDataVector) names += cellData->getName() + " ";
            nodePCellData.addAttribute("Scalars", names);
        }

        std::vector<XMLNode> nodePieceVector;
        for (int i = 0; i < this->getSize(); i++) {
            filename = filenameBase + "_%04i.vtu";
            snprintf(str_buffer, 256, filename.c_str(), i);
            filename = std::string(str_buffer);
            XMLNode nodePiece("Piece");
            nodePiece.addAttribute("Source", filename);
            nodePieceVector.push_back(nodePiece);
        }

        // Build structure
        root_.addChild(nodePUnstructuredGrid);
            nodePUnstructuredGrid.addChild(nodePPoints);
                nodePPoints.addChild(mesh.getPoints().toPVTK());
            nodePUnstructuredGrid.addChild(nodePPointData);
                for (auto& pointData : pointDataVector) nodePPointData.addChild(pointData->toPVTK());
            nodePUnstructuredGrid.addChild(nodePCellData);
                for (auto& cellData : cellDataVector) nodePCellData.addChild(cellData->toPVTK());
            for (auto& nodePiece : nodePieceVector) nodePUnstructuredGrid.addChild(nodePiece);
        
        // Write to file
        XMLTree tree(root_);
        tree.write(filenameBase + ".pvtu");

    }

}

} // NAMESPACE : EllipticForest