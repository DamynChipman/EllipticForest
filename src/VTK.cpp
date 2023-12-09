#include "VTK.hpp"

namespace EllipticForest {

XMLNode* DataArrayNodeBase::toVTK() {
    XMLNode* node_data_array = new XMLNode("DataArray");
    node_data_array->addAttribute("type", getType());
    node_data_array->addAttribute("Name", getName());
    node_data_array->addAttribute("NumberOfComponents", getNumberOfComponents());
    node_data_array->addAttribute("format", getFormat());
    node_data_array->addAttribute("RangeMin", getRangeMin());
    node_data_array->addAttribute("RangeMax", getRangeMax());
    node_data_array->data = getData();
    return node_data_array;
}

XMLNode* DataArrayNodeBase::toPVTK() {
    XMLNode* node_data_array = new XMLNode("PDataArray");
    node_data_array->addAttribute("type", getType());
    node_data_array->addAttribute("Name", getName());
    node_data_array->addAttribute("NumberOfComponents", getNumberOfComponents());
    node_data_array->addAttribute("format", getFormat());
    node_data_array->data = "";
    return node_data_array;
}

EmptyDataArrayNode::EmptyDataArrayNode() {}
std::string EmptyDataArrayNode::getType() { return "Float32"; }
std::string EmptyDataArrayNode::getName() { return "Empty"; }
std::string EmptyDataArrayNode::getNumberOfComponents() { return "1"; }
std::string EmptyDataArrayNode::getFormat() { return "ascii"; }
std::string EmptyDataArrayNode::getRangeMin() { return "0.0"; }
std::string EmptyDataArrayNode::getRangeMax() { return "0.0"; }
std::string EmptyDataArrayNode::getData() { return "0.0"; }

RectilinearGridVTK::RectilinearGridVTK() : mesh_complete_(false), coords_data_vector_(0), point_data_vector_(0), cell_data_vector_(0), root_("VTKFile") {
    root_.addAttribute("type", "RectilinearGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

void RectilinearGridVTK::buildMesh(RectilinearGridNodeBase& mesh, DataArrayNodeBase* x_coords, DataArrayNodeBase* y_coords, DataArrayNodeBase* z_coords) {
    mesh_ = &mesh;
    
    if (x_coords != nullptr) {
        coords_data_vector_.push_back(x_coords);
    }
    else {
        coords_data_vector_.push_back(&empty_data_array_);
    }

    if (y_coords != nullptr) {
        coords_data_vector_.push_back(y_coords);
    }
    else {
        coords_data_vector_.push_back(&empty_data_array_);
    }

    if (z_coords != nullptr) {
        coords_data_vector_.push_back(z_coords);
    }
    else {
        coords_data_vector_.push_back(&empty_data_array_);
    }

    mesh_complete_ = true;
    return;
}

void RectilinearGridVTK::addPointData(DataArrayNodeBase& point_data) {
    point_data_vector_.push_back(&point_data);
}

void RectilinearGridVTK::addCellData(DataArrayNodeBase& cell_data) {
    cell_data_vector_.push_back(&cell_data);
}

void RectilinearGridVTK::toVTK(std::string filename) {

    if (mesh_complete_) {

        XMLNode node_rectilinear_grid("RectilinearGrid");
        node_rectilinear_grid.addAttribute("WholeExtent", mesh_->getWholeExtent());

        XMLNode node_piece("Piece");
        node_piece.addAttribute("Extent", mesh_->getExtent());

        XMLNode node_cell_data("CellData");
        if (!cell_data_vector_.empty()) {
            std::string names = "";
            for (auto& cell_data : cell_data_vector_) names += cell_data->getName() + " ";
            node_cell_data.addAttribute("Scalars", names);
        }

        XMLNode node_point_data("PointData");
        if (!point_data_vector_.empty()) {
            std::string names = "";
            for (auto& point_data : point_data_vector_) names += point_data->getName() + " ";
            node_point_data.addAttribute("Scalars", names);
        }

        XMLNode node_coordinates("Coordinates");
        
        root_.addChild(node_rectilinear_grid);
        node_rectilinear_grid.addChild(node_piece);
        if (!cell_data_vector_.empty()) node_piece.addChild(node_cell_data);
        if (!point_data_vector_.empty()) node_piece.addChild(node_point_data);
        node_piece.addChild(node_coordinates);
        for (auto cell_data : cell_data_vector_) node_cell_data.addChild(cell_data->toVTK());
        for (auto point_data : point_data_vector_) node_point_data.addChild(point_data->toVTK());
        for (auto coordData : coords_data_vector_) node_coordinates.addChild(coordData->toVTK());

        XMLTree tree(root_);
        tree.write(filename);

    }
    else {
        throw std::invalid_argument("[RectilinearGridVTK::toVTK] mesh and data not complete.");
    }

}

UnstructuredGridVTK::UnstructuredGridVTK() :
    mesh_complete_(false),
    point_data_vector_(0),
    cell_data_vector_(0),
    root_("VTKFile") {
    root_.addAttribute("type", "UnstructuredGrid");
    root_.addAttribute("version", "0.1");
    root_.addAttribute("byte_order", "LittleEndian");
}

UnstructuredGridNodeBase& UnstructuredGridVTK::mesh() {
    return *mesh_;
}

std::vector<DataArrayNodeBase*>& UnstructuredGridVTK::pointDataVector() {
    return point_data_vector_;
}

std::vector<DataArrayNodeBase*>& UnstructuredGridVTK::cellDataVector() {
    return cell_data_vector_;
}

void UnstructuredGridVTK::buildMesh(UnstructuredGridNodeBase& mesh) {

    mesh_ = &mesh;
    mesh_complete_ = true;

}

void UnstructuredGridVTK::addPointData(DataArrayNodeBase& point_data) {
    point_data_vector_.push_back(&point_data);
}

void UnstructuredGridVTK::addCellData(DataArrayNodeBase& cell_data) {
    cell_data_vector_.push_back(&cell_data);
}

void UnstructuredGridVTK::toVTK(std::string filename) {

    if (mesh_complete_) {

        XMLNode node_unstructured_grid("UnstructuredGrid");

        XMLNode node_piece("Piece");
        node_piece.addAttribute("NumberOfPoints", mesh_->getNumberOfPoints());
        node_piece.addAttribute("NumberOfCells", mesh_->getNumberOfCells());

        XMLNode node_points("Points");
        XMLNode node_cells("Cells");

        XMLNode node_cell_data("CellData");
        if (!cell_data_vector_.empty()) {
            std::string names = "";
            for (auto& cell_data : cell_data_vector_) names += cell_data->getName() + " ";
            node_cell_data.addAttribute("Scalars", names);
        }

        XMLNode node_point_data("PointData");
        if (!point_data_vector_.empty()) {
            std::string names = "";
            for (auto& point_data : point_data_vector_) names += point_data->getName() + " ";
            node_point_data.addAttribute("Scalars", names);
        }

        root_.addChild(node_unstructured_grid);
            node_unstructured_grid.addChild(node_piece);
                node_piece.addChild(node_points);
                    node_points.addChild(mesh_->getPoints().toVTK());
                node_piece.addChild(node_cells);
                    node_cells.addChild(mesh_->getConnectivity().toVTK());
                    node_cells.addChild(mesh_->getOffsets().toVTK());
                    node_cells.addChild(mesh_->getTypes().toVTK());
                if (!cell_data_vector_.empty()) node_piece.addChild(node_cell_data);
                    for (auto& cell_data : cell_data_vector_) node_cell_data.addChild(cell_data->toVTK());
                if (!point_data_vector_.empty()) node_piece.addChild(node_point_data);
                    for (auto& point_data : point_data_vector_) node_point_data.addChild(point_data->toVTK());

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

void PUnstructuredGridVTK::addPointData(DataArrayNodeBase& point_data) {
    vtu.addPointData(point_data);
}

void PUnstructuredGridVTK::addCellData(DataArrayNodeBase& cell_data) {
    vtu.addCellData(cell_data);
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
        auto& point_data_vector = vtu.pointDataVector();
        auto& cell_data_vector = vtu.cellDataVector();

        // Build nodes
        XMLNode node_p_unstructured_grid("PUnstructuredGrid");
        node_p_unstructured_grid.addAttribute("GhostLevel", "0");

        XMLNode node_p_points("PPoints");
        
        XMLNode node_p_pointData("PPointData");
        if (!point_data_vector.empty()) {
            std::string names = "";
            for (auto& point_data : point_data_vector) names += point_data->getName() + " ";
            node_p_pointData.addAttribute("Scalars", names);
        }

        XMLNode node_p_cell_data("PCellData");
        if (!cell_data_vector.empty()) {
            std::string names = "";
            for (auto& cell_data : cell_data_vector) names += cell_data->getName() + " ";
            node_p_cell_data.addAttribute("Scalars", names);
        }

        std::vector<XMLNode> node_piece_vector;
        for (int i = 0; i < this->getSize(); i++) {
            filename = filenameBase + "_%04i.vtu";
            snprintf(str_buffer, 256, filename.c_str(), i);
            filename = std::string(str_buffer);
            XMLNode node_piece("Piece");
            node_piece.addAttribute("Source", filename);
            node_piece_vector.push_back(node_piece);
        }

        // Build structure
        root_.addChild(node_p_unstructured_grid);
            node_p_unstructured_grid.addChild(node_p_points);
                node_p_points.addChild(mesh.getPoints().toPVTK());
            node_p_unstructured_grid.addChild(node_p_pointData);
                for (auto& point_data : point_data_vector) node_p_pointData.addChild(point_data->toPVTK());
            node_p_unstructured_grid.addChild(node_p_cell_data);
                for (auto& cell_data : cell_data_vector) node_p_cell_data.addChild(cell_data->toPVTK());
            for (auto& node_piece : node_piece_vector) node_p_unstructured_grid.addChild(node_piece);
        
        // Write to file
        XMLTree tree(root_);
        tree.write(filenameBase + ".pvtu");

    }

}

} // NAMESPACE : EllipticForest