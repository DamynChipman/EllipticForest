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

} // NAMESPACE : EllipticForest