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

class PRectilinearGridVTK {



};

} // NAMESPACE : EllipticForest

#endif // VTK_HPP_