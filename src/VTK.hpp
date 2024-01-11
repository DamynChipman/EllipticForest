#ifndef VTK_HPP_
#define VTK_HPP_

#include <string>
#include <vector>

#include "XMLTree.hpp"
#include "MPI.hpp"

namespace EllipticForest {

class DataArrayNodeBase {

public:

    /**
     * @brief Get the data type of the array
     * 
     * From the VTK Legacy Format document:
     * "The data type of a single component of the array. This is one of Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64."
     * 
     * @return std::string 
     */
    virtual std::string getType() = 0;

    /**
     * @brief Get the name of the array
     * 
     * From the VTK Legacy Format document:
     * "The name of the array. This is usually a brief description of the data stored in the array."
     * 
     * @return std::string 
     */
    virtual std::string getName() = 0;

    /**
     * @brief Get the number of components per entry in the array
     * 
     * From the VTK Legacy Format document:
     * "The number of components per value in the array."
     * 
     * @return std::string 
     */
    virtual std::string getNumberOfComponents() = 0;

    /**
     * @brief Get the format of the data
     * 
     * From the VTK Legacy Format document:
     * "The means by which the data values themselves are stored in the file. This is “ascii”, “binary”, or “appended”."
     * 
     * @return std::string 
     */
    virtual std::string getFormat() = 0;

    /**
     * @brief Get the minimum of the data
     * 
     * @return std::string 
     */
    virtual std::string getRangeMin() = 0;

    /**
     * @brief Get the maximum of the data
     * 
     * @return std::string 
     */
    virtual std::string getRangeMax() = 0;

    /**
     * @brief Get the data
     * 
     * The data is written as a white-space delimited list of numbers.
     * 
     * @return std::string 
     */
    virtual std::string getData() = 0;

    /**
     * @brief Writes the data array to an XML node for a VTK file
     * 
     * @return XMLNode* 
     */
    XMLNode* toVTK();

    /**
     * @brief Writes the data array to an XML node for a PVTK file
     * 
     * @return XMLNode* 
     */
    XMLNode* toPVTK();

};

struct EmptyDataArrayNode : public DataArrayNodeBase {

    /**
     * @brief Construct a new EmptyDataArrayNode object (default)
     * 
     */
    EmptyDataArrayNode();

    /**
     * @brief Get the data type of the array
     * 
     * @return std::string 
     */
    std::string getType();

    /**
     * @brief Get the name of the data array
     * 
     * @return std::string 
     */
    std::string getName();

    /**
     * @brief Get the number of components per entry in the data array
     * 
     * @return std::string 
     */
    std::string getNumberOfComponents();

    /**
     * @brief Get the format of the data
     * 
     * @return std::string 
     */
    std::string getFormat();

    /**
     * @brief Get the minimum of the data
     * 
     * @return std::string 
     */
    std::string getRangeMin();

    /**
     * @brief Get the maximum of the data
     * 
     * @return std::string 
     */
    std::string getRangeMax();

    /**
     * @brief Get the data
     * 
     * @return std::string 
     */
    std::string getData();
};

class RectilinearGridNodeBase {

public:

    /**
     * @brief Get the whole extent of the grid
     * 
     * @return std::string 
     */
    virtual std::string getWholeExtent() = 0;

    /**
     * @brief Get the extent of this piece of the grid
     * 
     * @return std::string 
     */
    virtual std::string getExtent() = 0;

};

class RectilinearGridVTK {

public:

    /**
     * @brief Construct a new RectilinearGridVTK object (default)
     * 
     */
    RectilinearGridVTK();

    /**
     * @brief Build the mesh from a collection of x-, y-, and z-coordinates
     * 
     * @param mesh Rectilinear grid mesh reference
     * @param x_coords Data array of x-coordinates
     * @param y_coords Data array of y-coordinates
     * @param z_coords Data array of z-coordinates
     */
    void buildMesh(RectilinearGridNodeBase& mesh, DataArrayNodeBase* x_coords, DataArrayNodeBase* y_coords, DataArrayNodeBase* z_coords);

    /**
     * @brief Adds PointData to the mesh
     * 
     * @param point_data Data array of point data
     */
    void addPointData(DataArrayNodeBase& point_data);

    /**
     * @brief Adds CellData to the mesh
     * 
     * @param cell_data Data array of cell data
     */
    void addCellData(DataArrayNodeBase& cell_data);

    /**
     * @brief Writes the mesh to VTK
     * 
     * @param filename 
     */
    void toVTK(std::string filename);

private:

    /**
     * @brief Flag to check if the mesh is complete
     * 
     */
    bool mesh_complete_;

    /**
     * @brief Root node of the XML tree
     * 
     */
    XMLNode root_;

    /**
     * @brief Pointer to the rectilinear mesh
     * 
     */
    RectilinearGridNodeBase* mesh_;

    /**
     * @brief Default data array for unused pieces of the mesh
     * 
     */
    EmptyDataArrayNode empty_data_array_{};

    /**
     * @brief Vector of coordinates
     * 
     */
    std::vector<DataArrayNodeBase*> coords_data_vector_;

    /**
     * @brief Vector of points
     * 
     */
    std::vector<DataArrayNodeBase*> point_data_vector_;

    /**
     * @brief Vector of cells
     * 
     */
    std::vector<DataArrayNodeBase*> cell_data_vector_;

};

class UnstructuredGridNodeBase {

public:

    /**
     * @brief Get the number of points in the mesh
     * 
     * @return std::string 
     */
    virtual std::string getNumberOfPoints() = 0;

    /**
     * @brief Get the number of cells in the mesh
     * 
     * @return std::string 
     */
    virtual std::string getNumberOfCells() = 0;

    /**
     * @brief Get the data array of points
     * 
     * @return DataArrayNodeBase& 
     */
    virtual DataArrayNodeBase& getPoints() = 0;

    /**
     * @brief Get the data array of connectivity data
     * 
     * @return DataArrayNodeBase& 
     */
    virtual DataArrayNodeBase& getConnectivity() = 0;

    /**
     * @brief Get the data array of offset data
     * 
     * @return DataArrayNodeBase& 
     */
    virtual DataArrayNodeBase& getOffsets() = 0;
    
    /**
     * @brief Get the data array of cell types
     * 
     * @return DataArrayNodeBase& 
     */
    virtual DataArrayNodeBase& getTypes() = 0;

};

class UnstructuredGridVTK {

public:

    /**
     * @brief Construct a new UnstructuredGridVTK object (default)
     * 
     */
    UnstructuredGridVTK();
    
    /**
     * @brief Returns the unstructured mesh reference
     * 
     * @return UnstructuredGridNodeBase& 
     */
    UnstructuredGridNodeBase& mesh();

    /**
     * @brief Returns the vector of point data arrays
     * 
     * @return std::vector<DataArrayNodeBase*>& 
     */
    std::vector<DataArrayNodeBase*>& pointDataVector();

    /**
     * @brief Returns the vector of cell data arrays
     * 
     * @return std::vector<DataArrayNodeBase*>& 
     */
    std::vector<DataArrayNodeBase*>& cellDataVector();

    /**
     * @brief Builds the unstructured mesh
     * 
     * @param mesh Reference to implemented mesh instance
     */
    void buildMesh(UnstructuredGridNodeBase& mesh);

    /**
     * @brief Add point data to the mesh
     * 
     * @param point_data Data array of point data
     */
    void addPointData(DataArrayNodeBase& point_data);

    /**
     * @brief Add cell data to the mesh
     * 
     * @param cell_data Data array of cell data
     */
    void addCellData(DataArrayNodeBase& cell_data);

    /**
     * @brief Writes to a VTK file
     * 
     * @param filename Basename of VTK file
     */
    void toVTK(std::string filename);


private:

    /**
     * @brief Flag to check if mesh is built
     * 
     */
    bool mesh_complete_;

    /**
     * @brief Number of points in the mesh
     * 
     */
    int npoints_;

    /**
     * @brief Number of cells in the mesh
     * 
     */
    int ncells_;

    /**
     * @brief Root of the XML tree
     * 
     */
    XMLNode root_;

    /**
     * @brief Pointer to the unstructured mesh
     * 
     */
    UnstructuredGridNodeBase* mesh_;

    /**
     * @brief Vector of point data arrays
     * 
     */
    std::vector<DataArrayNodeBase*> point_data_vector_;

    /**
     * @brief Vector of cell data arrays
     * 
     */
    std::vector<DataArrayNodeBase*> cell_data_vector_;

};

class PUnstructuredGridVTK : public MPI::MPIObject {

public:

    /**
     * @brief The rank local version of the unstructured mesh
     * 
     */
    UnstructuredGridVTK vtu{};

    /**
     * @brief Construct a new PUnstructuredGridVTK object (default)
     * 
     */
    PUnstructuredGridVTK();

    /**
     * @brief Construct a new PUnstructuredGridVTK object on a communicator
     * 
     * @param comm MPI communicator
     */
    PUnstructuredGridVTK(MPI_Comm comm);

    /**
     * @brief Builds the unstructured mesh in parallel
     * 
     * @param mesh Reference to implemented mesh instance
     */
    void buildMesh(UnstructuredGridNodeBase& mesh);

    /**
     * @brief Add point data to the mesh
     * 
     * @param point_data Data array of point data
     */
    void addPointData(DataArrayNodeBase& point_data);

    /**
     * @brief Add cell data to the mesh
     * 
     * @param cell_data Data array of cell data
     */
    void addCellData(DataArrayNodeBase& cell_data);

    /**
     * @brief Writes the mesh to a VTK file in parallel
     * 
     * @param filename_base Base filename
     */
    void toVTK(std::string filename_base);

private:

    XMLNode root_;

};

} // NAMESPACE : EllipticForest

#endif // VTK_HPP_