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

    Quadtree<PatchType> quadtree;
    // PatchType* root_patch;
    // AbstractNodeFactory<PatchType>* node_factory;
    int nPoints = 0;
    int nCells = 0;
    Vector<double> points{};
    Vector<int> connectivity{};
    Vector<int> offsets{};
    Vector<int> types{};
    std::vector<Vector<double>*> meshFunctions{};
    // void* utility_pointer;

    Mesh() :
        MPIObject{MPI_COMM_WORLD}
            {}

    Mesh(Quadtree<PatchType> quadtree) :
        MPIObject{quadtree.getComm()},
        quadtree{quadtree} {
        
        setMeshFromQuadtree();

    }

    Mesh(MPI_Comm comm, p4est_t* p4est, PatchType& rootData, AbstractNodeFactory<PatchType>& nodeFactory) :
        MPIObject{comm},
        quadtree{comm, p4est, rootData, nodeFactory} {
    
        setMeshFromQuadtree();
    
    }

    void refineByFunction(std::function<bool(double x, double y)> fn, double threshold, int min_level, int max_level, PatchType& root_patch, AbstractNodeFactory<PatchType>& node_factory) {
        // Create p4est to create quadtree
        auto& root_grid = root_patch.grid();
        double xlower = root_grid.xLower();
        double xupper = root_grid.xUpper();
        double ylower = root_grid.yLower();
        double yupper = root_grid.yUpper();
        p4est_connectivity_t* conn = p4est::p4est_connectivity_new_square_domain(xlower, xupper, ylower, yupper);
        p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, min_level, true, 0, NULL, NULL);
        p4est->user_pointer = this;

        refine_fn_ = fn;
        threshold_ = threshold;
        nx_ = root_grid.nx();
        ny_ = root_grid.ny();
        min_level_ = min_level;
        max_level_ = max_level;

        // Refine quadtree
        p4est_refine(
            p4est,
            1,
            [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
                auto* mesh = (Mesh<PatchType>*) p4est->user_pointer;

                if (quadrant->level >= mesh->max_level_) {
                    return 0;
                }

                double vxyz[3];
                double xlower, xupper, ylower, yupper;
                p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
                xlower = vxyz[0];
                ylower = vxyz[1];

                p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
                xupper = vxyz[0];
                yupper = vxyz[1];

                double dx = (xupper - xlower) / (mesh->nx_);
                double dy = (yupper - ylower) / (mesh->ny_);
                for (int i = 0; i < mesh->nx_; i++) {
                    for (int j = 0; j < mesh->ny_; j++) {
                        double x = xlower + (i + 0.5)*dx;
                        double y = ylower + (j + 0.5)*dy;
                        if (mesh->refine_fn_(x,y)) {
                            return 1;
                        }
                    }
                }

                return 0;
            },
            NULL
        ); // p4est_refine

        // Balance and partition
        p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);
        p4est_partition(p4est, 0, NULL);

        // Create quadtree
        quadtree = Quadtree<PatchType>{MPI_COMM_WORLD, p4est, root_patch, node_factory};

        // Create mesh
        setMeshFromQuadtree();

        return;
    }

    void setMeshFromQuadtree() {

        points.name() = "points";
        points.setNumberOfComponents("3");
        connectivity.name() = "connectivity";
        connectivity.setType("Int64");
        offsets.name() = "offsets";
        offsets.setType("Int64");
        types.name() = "types";
        types.setType("Int64");

        int index = 0;

        quadtree.traversePreOrder([&](Node<PatchType>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                double xLower = grid.xLower();
                double xUpper = grid.xUpper();
                double yLower = grid.yLower();
                double yUpper = grid.yUpper();
                double dx = grid.dx();
                double dy = grid.dy();
                int nx = grid.nx();
                int ny = grid.ny();

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

    void iteratePatches(std::function<void(PatchType& patch)> fn) {
        quadtree.traversePreOrder([&](Node<PatchType>* node){
            if (node->leaf) {
                auto& patch = node->data;
                fn(patch);
            }
            return 1;
        });
    }

    void iterateCells(std::function<void(double xc, double yc)> fn) {
        quadtree.traversePreOrder([&](Node<PatchType>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                int nx = grid.nx();
                int ny = grid.ny();

                for (int i = 0; i < nx; i++) {
                    double xc = grid(0, i);
                    for (int j = 0; j < ny; j++) {
                        double yc = grid(1, j);
                        fn(xc, yc);
                    }
                }
            }
            return 1;
        });
    }

    // void refineByFunction(std::function<bool(double x, double y, double threshold) fn, bool recursiveRefinement) {
    //     // NOTE: Working here!

    //     p4est_refine(quadtree.p4est, recursiveRefinement,
    //     [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

    //     });
    // }

    void addMeshFunction(Vector<double>& meshFunction) { meshFunctions.push_back(&meshFunction); }

    void addMeshFunction(std::function<double(double x, double y)> meshFunction, std::string meshFunctionName) {

        std::size_t nCells = 0;
        iteratePatches([&](PatchType& patch){
            auto& grid = patch.grid();
            nCells += grid.nx() * grid.ny();
        });

        Vector<double>* meshFunctionVector = new Vector<double>(nCells);
        meshFunctionVector->name() = meshFunctionName;
        int i = 0;
        iterateCells([&](double xc, double yc){
            (*meshFunctionVector)[i++] = meshFunction(xc, yc);
        });

        meshFunctions.push_back(meshFunctionVector);

    }

    void clearMeshFunctions() {
        meshFunctions.clear();
    }

    void toVTK(std::string baseFilename, int number=-1) {
        // Output mesh
        std::string meshFilename = baseFilename + "-mesh";
        if (number != -1) {
            char buffer[32];
            snprintf(buffer, 32, "%04i", number);
            meshFilename += "-" + std::string(buffer);
        }
        PUnstructuredGridVTK pvtu{};
        pvtu.buildMesh(*this);
        for (auto& meshFunction : meshFunctions) {
            pvtu.addCellData(*meshFunction);
        }
        pvtu.toVTK(meshFilename);

        // Output quadtree (p4est)
        std::string quadtreeFilename = baseFilename + "-quadtree";
        if (number != -1) {
            char buffer[32];
            snprintf(buffer, 32, "%04i", number);
            quadtreeFilename += "-" + std::string(buffer);
        }
        p4est_vtk_context_t* vtk_context = p4est_vtk_context_new(quadtree.p4est, quadtreeFilename.c_str());
        p4est_vtk_context_set_scale(vtk_context, 1.0);
        vtk_context = p4est_vtk_write_header(vtk_context);
        p4est_vtk_write_footer(vtk_context);
    }

private:

    // Variables to pass to p4est_refine via this pointer
    std::function<bool(double x, double y)> refine_fn_;
    double threshold_;
    double x_lower_;
    double x_upper_;
    double y_lower_;
    double y_upper_;
    int nx_;
    int ny_;
    int min_level_;
    int max_level_;

};
    
} // NAMESPACE : EllipticForest

#endif // MESH_HPP_