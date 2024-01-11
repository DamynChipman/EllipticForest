#include <iostream>
#include <string>
#include <vector>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

using namespace EllipticForest;

/**
 * @brief User-defined solution function
 * 
 * When computing error, calls this function to compare with numerical solution.
 * 
 * Used by EllipticForest to create the boundary data.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double uFunction(double x, double y) {
    return sin(x) + sin(y);
}

/**
 * @brief User-defined load function
 * 
 * Right-hand side function of the elliptic PDE.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double fFunction(double x, double y) {
    return -uFunction(x, y);
}

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_SELF);

    bool vtk_flag = true;
    double threshold = 1.2;
    int min_level = 2;
    int max_level = 4;
    double x_lower = -10;
    double x_upper = 10;
    double y_lower = -10;
    double y_upper = 10;
    int nx = 16;
    int ny = 16;

    FiniteVolumeGrid grid(mpi.getComm(), nx, x_lower, x_upper, ny, y_lower, y_upper);
    FiniteVolumePatch root_patch(mpi.getComm(), grid);

    FiniteVolumeSolver solver{};
    solver.solver_type = FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = [&](double x, double y){
        return (double) 1;
    };
    solver.beta_function = [&](double x, double y){
        return (double) 1;
    };
    solver.lambda_function = [&](double x, double y){
        return (double) 0;
    };
    solver.rhs_function = [&](double x, double y){
        return fFunction(x, y);
    };

    FiniteVolumeNodeFactory node_factory(mpi.getComm(), solver);
    Quadtree<FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory);
    quadtree.refine(true,
        [&](Node<FiniteVolumePatch>* node){
            return (int) node->level < min_level;
        }
    );
    Mesh<FiniteVolumePatch> mesh(quadtree);
    // Mesh<FiniteVolumePatch> mesh{};
    // mesh.refineByFunction(
    //     [&](double x, double y){
    //         double f = fFunction(x, y);
    //         return fabs(f) > threshold;
    //     },
    //     threshold,
    //     min_level,
    //     max_level,
    //     root_patch,
    //     node_factory
    // );


    HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double> HPS(mpi.getComm(), mesh, solver);
    HPS.setupStage();
    HPS.buildStage();
    HPS.upwardsStage([&](double x, double y){
        return fFunction(x, y);
    });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });

    if (vtk_flag) {
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        
        EllipticForest::Vector<double> fMesh{};
        fMesh.name() = "f_rhs";
        
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
                fMesh.append(patch.vectorF());
            }
            return 1;
        });

        // Add mesh functions to mesh
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction(fMesh);
        mesh.addMeshFunction(
            [&](double x, double y){
                return uFunction(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic-step0");
    }

    std::vector<std::string> paths = {"000", "033"};
    for (auto path : paths) {
        mesh.quadtree.refineNode(path, true);
    }

    mesh.quadtree.merge(
        [&](Node<FiniteVolumePatch>* patch){
            return 1;
        },
        [&](Node<FiniteVolumePatch>* parent_node, std::vector<Node<FiniteVolumePatch>*> children_nodes){
            FiniteVolumeHPS::coarsen(parent_node->data, children_nodes[0]->data, children_nodes[1]->data, children_nodes[2]->data, children_nodes[3]->data);
            return 1;
        }
    );
    HPS.upwardsStage([&](double x, double y){
        return fFunction(x, y);
    });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });

    mesh.clear();
    mesh.setMeshFromQuadtree();
    if (vtk_flag) {
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        
        EllipticForest::Vector<double> fMesh{};
        fMesh.name() = "f_rhs";
        
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
                fMesh.append(patch.vectorF());
            }
            return 1;
        });

        // Add mesh functions to mesh
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction(fMesh);
        mesh.addMeshFunction(
            [&](double x, double y){
                return uFunction(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic-step1");
    }

    for (auto path : paths) {
        mesh.quadtree.coarsenNode(path, true);
    }

    HPS.upwardsStage([&](double x, double y){
        return fFunction(x, y);
    });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });

    mesh.clear();
    mesh.setMeshFromQuadtree();
    if (vtk_flag) {
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        
        EllipticForest::Vector<double> fMesh{};
        fMesh.name() = "f_rhs";
        
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
                fMesh.append(patch.vectorF());
            }
            return 1;
        });

        // Add mesh functions to mesh
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction(fMesh);
        mesh.addMeshFunction(
            [&](double x, double y){
                return uFunction(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic-step2");
    }

    return EXIT_SUCCESS;
}