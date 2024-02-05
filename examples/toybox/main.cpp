#include <iostream>
#include <string>
#include <vector>
#include <random>

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

/**
 * @brief User-defined alpha function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double alphaFunction(double x, double y) {
    return 1.0;
}

/**
 * @brief User-defined beta function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double betaFunction(double x, double y) {
    return 1.0;
}

/**
 * @brief User-defined lambda function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double lambdaFunction(double x, double y) {
    return 0.0;
}

int randomIntInRange(int min, int max) {
    static std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double computeMaxError(Quadtree<FiniteVolumePatch>& quadtree) {
    double max_abs_error = 0.;
    quadtree.traversePreOrder([&](Node<FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();
            for (int i = 0; i < grid.nx(); i++) {
                for (int j = 0; j < grid.ny(); j++) {
                    double xc = grid(0, i);
                    double yc = grid(1, j);
                    int index = j + i*grid.ny();
                    double u_exact = uFunction(xc, yc);
                    double u_approx = patch.vectorU()[index];
                    double abs_error = abs(u_exact - u_approx);
                    max_abs_error = std::max(max_abs_error, abs_error);
                }
            }
        }
        return 1;
    });
    return max_abs_error;
}

double computeL2Error(Quadtree<FiniteVolumePatch>& quadtree) {
    double l2_error = 0;
    quadtree.traversePreOrder([&](Node<FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();
            for (int i = 0; i < grid.nx(); i++) {
                for (int j = 0; j < grid.ny(); j++) {
                    double xc = grid(0, i);
                    double yc = grid(1, j);
                    int index = j + i*grid.ny();
                    double u_exact = uFunction(xc, yc);
                    double u_approx = patch.vectorU()[index];
                    l2_error += pow(u_exact - u_approx, 2);
                }
            }
        }
        return 1;
    });
    return sqrt(l2_error);
}

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_SELF);

    // bool vtk_flag = true;
    // double threshold = 1.2;
    // int min_level = 4;
    // int max_level = 8;
    // double x_lower = -10;
    // double x_upper = 10;
    // double y_lower = -10;
    // double y_upper = 10;
    // int nx = 16;
    // int ny = 16;

    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);
    
    int min_level = 4;
    app.options.setOption("min-level", min_level);
    
    int max_level = 8;
    app.options.setOption("max-level", max_level);

    double x_lower = -10.0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 10.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = -10.0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 10.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 16;
    app.options.setOption("nx", nx);
    
    int ny = 16;
    app.options.setOption("ny", ny);

    double threshold = 1.2;
    app.options.setOption("threshold", threshold);

    FiniteVolumeGrid grid(mpi.getComm(), nx, x_lower, x_upper, ny, y_lower, y_upper);
    FiniteVolumePatch root_patch(mpi.getComm(), grid);

    FiniteVolumeSolver solver{};
    solver.solver_type = FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

    FiniteVolumeNodeFactory node_factory(mpi.getComm(), solver);
    Mesh<FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            double f = fFunction(x, y);
            return fabs(f) > threshold;
        },
        threshold,
        min_level,
        min_level,
        root_patch,
        node_factory
    );
    // Quadtree<FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory, {x_lower, x_upper, y_lower, y_upper});
    // Mesh<FiniteVolumePatch> mesh(quadtree);
    // mesh.quadtree.refine(true,
    //     [&](Node<FiniteVolumePatch>* node){
    //         return (int) node->level < min_level;
    //     }
    // );

    HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double> HPS(mpi.getComm(), mesh, solver);
    HPS.setupStage();
    HPS.buildStage();

    int n;
    int n_solves = 100;
    int file_counter = 0;
    double max_error;
    double l2_error;
    std::vector<std::string> adapt_paths(n_solves);
    app.log("==================== Begin refinement ====================");
    for (n = 1; n <= n_solves; n++) {
        // Solve
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });

        // Error
        max_error = computeMaxError(mesh.quadtree);
        l2_error = computeL2Error(mesh.quadtree);

        // Print
        app.log("Solve # " + std::to_string(n) + " of " + std::to_string(n_solves) + " - L2 Error = %16.8e, Max Error = %16.8e", l2_error, max_error);

        // Output
        mesh.clear();
        mesh.setMeshFromQuadtree();
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
            }
            return 1;
        });
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction([&](double x, double y){
            return uFunction(x, y);
        }, "u_exact");
        mesh.toVTK("toybox", file_counter++);

        // Refine
        if (n != n_solves) {
            int n_leaf_patches = 0;
            mesh.quadtree.traversePreOrder([&](Node<FiniteVolumePatch>* node){
                if (node->leaf) {
                    n_leaf_patches++;
                }
                return 1;
            });
            
            int id_to_refine = randomIntInRange(0, n_leaf_patches);

            int leaf_counter = 0;
            std::string path_to_refine = "";
            mesh.quadtree.traversePreOrder([&](Node<FiniteVolumePatch>* node){
                if (node->leaf) {
                    if (leaf_counter == id_to_refine) {
                        path_to_refine = node->path;
                    }
                    leaf_counter++;
                }
                return 1;
            });
            app.log("  n_leaf_patches = %i", n_leaf_patches);
            app.log("  id_to_refine = %i", id_to_refine);
            app.log("  path_to_refine = " + path_to_refine);
            adapt_paths[n] = path_to_refine;
            // app.log("Refining node " + path_to_refine);
            mesh.quadtree.refineNode(path_to_refine, true);
            // mesh.quadtree.balance(BalancePolicy::CORNER);
            mesh.quadtree.merge(
                [&](Node<FiniteVolumePatch>* patch){
                    return 1;
                },
                [&](Node<FiniteVolumePatch>* parent_node, std::vector<Node<FiniteVolumePatch>*> children_nodes){
                    FiniteVolumeHPS::coarsen(parent_node->data, children_nodes[0]->data, children_nodes[1]->data, children_nodes[2]->data, children_nodes[3]->data);
                    return 1;
                }
            );
        }

    }

    app.log("==================== Begin coarsening ====================");
    for (int n = n_solves; n >= 1; n--) {
        
        // Coarsen
        auto& path_to_coarsen = adapt_paths[n-1];
        app.log("  path_to_coarsen = " + path_to_coarsen);
        mesh.quadtree.coarsenNode(path_to_coarsen, true);
        // mesh.quadtree.balance(BalancePolicy::CORNER);
        mesh.quadtree.merge(
            [&](Node<FiniteVolumePatch>* patch){
                return 1;
            },
            [&](Node<FiniteVolumePatch>* parent_node, std::vector<Node<FiniteVolumePatch>*> children_nodes){
                FiniteVolumeHPS::coarsen(parent_node->data, children_nodes[0]->data, children_nodes[1]->data, children_nodes[2]->data, children_nodes[3]->data);
                return 1;
            }
        );

        // Solve
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });

        // Error
        max_error = computeMaxError(mesh.quadtree);
        l2_error = computeL2Error(mesh.quadtree);

        // Print
        app.log("Solve # " + std::to_string(n) + " of " + std::to_string(n_solves) + " - L2 Error = %16.8e, Max Error = %16.8e", l2_error, max_error);

        // Output
        mesh.clear();
        mesh.setMeshFromQuadtree();
        EllipticForest::Vector<double> uMesh{};
        uMesh.name() = "u_soln";
        mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
            }
            return 1;
        });
        mesh.addMeshFunction(uMesh);
        mesh.addMeshFunction([&](double x, double y){
            return uFunction(x, y);
        }, "u_exact");
        mesh.toVTK("toybox", file_counter++);

    }

    return EXIT_SUCCESS;
}