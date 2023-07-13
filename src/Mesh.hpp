#ifndef MESH_HPP_
#define MESH_HPP_

#include "Vector.hpp"
#include "Quadtree.hpp"
#include "MPI.hpp"
#include "VTK.hpp"

namespace EllipticForest {

template<typename PatchType>
class Mesh : public MPI::MPIObject, public UnstructuredGridNodeBase {

public:

    Quadtree<PatchType>* quadtree;
    int nPoints = 0;
    int nCells = 0;
    Vector<double> points{};
    Vector<int> connectivity{};
    Vector<int> offsets{};
    Vector<int> types{};

    Mesh() :
        MPIObject{MPI_COMM_WORLD},
        quadtree{nullptr} {
            init();
        }

    Mesh(Quadtree<PatchType>& quadtree) :
        MPIObject{quadtree.getComm()},
        quadtree{&quadtree} {
            init();
        }

    void init() {

        points.name() = "points";
        points.setNumberOfComponents("3");
        connectivity.name() = "connectivity";
        connectivity.setType("Int64");
        offsets.name() = "offsets";
        offsets.setType("Int64");
        types.name() = "types";
        types.setType("Int64");

        int index = 0;

        quadtree->traversePreOrder([&](Node<PatchType>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                double xLower = grid.xLower();
                double xUpper = grid.xUpper();
                double yLower = grid.yLower();
                double yUpper = grid.yUpper();
                double dx = grid.dx();
                double dy = grid.dy();
                int nx = grid.nPointsX();
                int ny = grid.nPointsY();

                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < ny; j++) {
                        double x0 = xLower + i*dx;
                        double y0 = yLower + j*dy;
                        double z0 = 0.0;

                        double x1 = xLower + (i+1)*dx;
                        double y1 = yLower + j*dy;
                        double z1 = 0.0;

                        double x2 = xLower + (i+1)*dx;
                        double y2 = yLower + (j+1)*dy;
                        double z2 = 0.0;

                        double x3 = xLower + i*dx;
                        double y3 = yLower + (j+1)*dy;
                        double z3 = 0.0;

                        Vector<double> cellCorners = {
                            x0, y0, z0,
                            x1, y1, z1,
                            x2, y2, z2,
                            x3, y3, z3
                        };
                        Vector<int> cellConnectivity = {
                            index + 0,
                            index + 1,
                            index + 2,
                            index + 3
                        };
                        Vector<int> cellOffsets = {
                            (index + 1) * 4,
                            (index + 2) * 4,
                            (index + 3) * 4,
                            (index + 4) * 4
                        };
                        Vector<int> cellTypes = {
                            9, 9, 9, 9
                        };
                        
                        index += 4;
                        nPoints += 4;
                        nCells += 1;

                        points.append(cellCorners);
                        connectivity.append(cellConnectivity);
                        offsets.append(cellOffsets);
                        types.append(cellTypes);
                    }
                }
            }
            return 1;
        });

    }

    std::string getNumberOfPoints() { return std::to_string(nPoints); }
    std::string getNumberOfCells() { return std::to_string(nCells); }
    DataArrayNodeBase& getPoints() { return points; }
    DataArrayNodeBase& getConnectivity() { return connectivity; }
    DataArrayNodeBase& getOffsets() { return offsets; }
    DataArrayNodeBase& getTypes() { return types; }

};
    
} // NAMESPACE : EllipticForest

#endif // MESH_HPP_