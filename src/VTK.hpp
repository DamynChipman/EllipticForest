#ifndef VTK_HPP_
#define VTK_HPP_

#include <string>
#include <vector>

#include "XMLTree.hpp"

namespace EllipticForest {

class DataArrayNodeBase {

public:

    virtual std::string getType() = 0;
    virtual std::string getName() = 0;
    virtual std::string getNumberOfComponents() = 0;
    virtual std::string getFormat() = 0;
    virtual std::string getRangeMin() = 0;
    virtual std::string getRangeMax() = 0;
    virtual std::string getData() = 0;

    XMLNode* toVTK();

};

struct EmptyDataArrayNode : public DataArrayNodeBase {
    EmptyDataArrayNode();
    std::string getType();
    std::string getName();
    std::string getNumberOfComponents();
    std::string getFormat();
    std::string getRangeMin();
    std::string getRangeMax();
    std::string getData();
};

class RectilinearGridNodeBase {

public:

    virtual std::string getWholeExtent() = 0;
    virtual std::string getExtent() = 0;

};

class RectilinearGridVTK {

public:

    RectilinearGridVTK();
    void buildMesh(RectilinearGridNodeBase& mesh, DataArrayNodeBase* xCoords, DataArrayNodeBase* yCoords, DataArrayNodeBase* zCoords);
    void addPointData(DataArrayNodeBase& pointData);
    void addCellData(DataArrayNodeBase& cellData);
    void toVTK(std::string filename);

private:

    bool meshComplete_;
    XMLNode root_;
    RectilinearGridNodeBase* mesh_;
    EmptyDataArrayNode emptyDataArray_{};
    std::vector<DataArrayNodeBase*> coordsDataVector_;
    std::vector<DataArrayNodeBase*> pointDataVector_;
    std::vector<DataArrayNodeBase*> cellDataVector_;

};

class UnstructuredGridNodeBase {

public:

    virtual std::string getNumberOfPoints() = 0;
    virtual std::string getNumberOfCells() = 0;
    virtual DataArrayNodeBase& getPoints() = 0;
    virtual DataArrayNodeBase& getConnectivity() = 0;
    virtual DataArrayNodeBase& getOffsets() = 0;
    virtual DataArrayNodeBase& getTypes() = 0;

};

class PRectilinearGridVTK {



};

class UnstructuredGridVTK {

public:

    UnstructuredGridVTK();

    void buildMesh(UnstructuredGridNodeBase& mesh);
    void addPointData(DataArrayNodeBase& pointData);
    void addCellData(DataArrayNodeBase& cellData);
    void toVTK(std::string filename);

private:

    bool meshComplete_;
    int npoints_;
    int ncells_;
    XMLNode root_;
    UnstructuredGridNodeBase* mesh_;
    std::vector<DataArrayNodeBase*> pointDataVector_;
    std::vector<DataArrayNodeBase*> cellDataVector_;

};

} // NAMESPACE : EllipticForest

#endif // VTK_HPP_