#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <SpecialMatrices.hpp>
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

double computeMaxErrorFromExact(Quadtree<FiniteVolumePatch>& quadtree) {
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
                    double abs_error = fabs(u_exact - u_approx);
                    max_abs_error = fmax(max_abs_error, abs_error);
                }
            }
        }
        return 1;
    });
    return max_abs_error;
}

double computeL2ErrorFromExact(Quadtree<FiniteVolumePatch>& quadtree) {
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
                    l2_error += (grid.dx()*grid.dy())*pow(fabs(u_exact - u_approx), 2);
                }
            }
        }
        return 1;
    });
    return sqrt(l2_error / pow(20., 2));
}

void writeMesh(EllipticForest::Mesh<EllipticForest::FiniteVolumePatch>& mesh, int n_output) {
    auto& app = EllipticForest::EllipticForestApp::getInstance();
    mesh.clear();
    mesh.setMeshFromQuadtree();
    app.logHead("Output mesh: %04i", n_output);
    
    // Extract out solution and right-hand side data stored on leaves
    EllipticForest::Vector<double> u_mesh{};
    u_mesh.name() = "u_soln";

    EllipticForest::Vector<double> u_diff{};
    u_diff.name() = "u_diff";
    
    mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();

            u_mesh.append(patch.vectorU());
            for (int i = 0; i < grid.nx(); i++) {
                for (int j = 0; j < grid.ny(); j++) {
                    double xc = grid(0, i);
                    double yc = grid(1, j);
                    int index = j + i*grid.ny();
                    u_diff.append(patch.vectorU()[index] - uFunction(xc, yc));
                }
            }
        }
        return 1;
    });

    // Add mesh functions to mesh
    mesh.addMeshFunction(u_mesh);
    mesh.addMeshFunction(u_diff);
    // mesh.addMeshFunction(fMesh);

    // Write VTK files:
    //      "toybox-mesh-{n}.pvtu"            : Parallel header file for mesh and data
    //      "toybox-quadtree-{n}.pvtu"        : p4est quadtree structure
    mesh.toVTK("toybox", n_output);
}

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_SELF);

    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);
    
    int min_level = 2;
    app.options.setOption("min-level", min_level);
    
    int max_level = 5;
    app.options.setOption("max-level", max_level);

    double x_lower = -10.0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 10.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = -10.0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 10.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 8;
    app.options.setOption("nx", nx);
    
    int ny = 8;
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
    EllipticForest::Quadtree<EllipticForest::FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory, {x_lower, x_upper, y_lower, y_upper});
    quadtree.refine(true,
        [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            if (node->level >= max_level) {
                return (int) false;
            }
            if (node->level <= min_level) {
                return (int) true;
            }
            auto& grid = node->data.grid();
            for (int i = 0; i < grid.nx(); i++) {
                for (int j = 0; j < grid.ny(); j++) {
                    double x = grid(0, i);
                    double y = grid(1, j);
                    if (fabs(fFunction(x, y)) > threshold) {
                        return (int) true;
                    }
                }
            }
            return (int) false;
        }
    );
    quadtree.balance(EllipticForest::BalancePolicy::CORNER);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh(quadtree);

    HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double> HPS(mpi.getComm(), mesh, solver);
    
    // Factorize; solve on initial mesh
    HPS.buildStage();
    HPS.upwardsStage([&](double x, double y){
        return fFunction(x, y);
    });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });
    double error_inf = computeMaxErrorFromExact(mesh.quadtree);
    double error_l2 = computeL2ErrorFromExact(mesh.quadtree);
    app.log("--=== Solve on Initial Mesh ===--");
    app.log("error-inf = %24.16e", error_inf);
    app.log("error-l2  = %24.16e", error_l2);
    writeMesh(mesh, 0);

    // Refine the mesh in random spots
    int n_adapts = 20;
    for (int n = 0; n < n_adapts; n++) {
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
        mesh.quadtree.refineNode(path_to_refine, true);
        mesh.quadtree.balance(EllipticForest::BalancePolicy::CORNER);
    }

    // Solve refined mesh w/o factorization
    // HPS.upwardsStage([&](double x, double y){
    //     return fFunction(x, y);
    // });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });
    error_inf = computeMaxErrorFromExact(mesh.quadtree);
    error_l2 = computeL2ErrorFromExact(mesh.quadtree);
    app.log("--=== Solve on Adapted Mesh ===--");
    app.log("error-inf = %24.16e", error_inf);
    app.log("error-l2  = %24.16e", error_l2);
    writeMesh(mesh, 1);
    
    EllipticForest::Vector<double> u_mesh_adapted{};
    mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            u_mesh_adapted.append(node->data.vectorU());
        }
        return 1;
    });

    // Rebuild and solve on new mesh
    HPS.buildStage();
    HPS.upwardsStage([&](double x, double y){
        return fFunction(x, y);
    });
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return uFunction(x, y);
    });
    error_inf = computeMaxErrorFromExact(mesh.quadtree);
    error_l2 = computeL2ErrorFromExact(mesh.quadtree);
    app.log("--=== Solve on Refactored Mesh ===--");
    app.log("error-inf = %24.16e", error_inf);
    app.log("error-l2  = %24.16e", error_l2);
    writeMesh(mesh, 2);

    EllipticForest::Vector<double> u_mesh_refactored{};
    mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            u_mesh_refactored.append(node->data.vectorU());
        }
        return 1;
    });

    auto u_mesh_diff = u_mesh_refactored - u_mesh_adapted;
    error_inf = *std::max_element(u_mesh_diff.data().begin(), u_mesh_diff.data().end());
    app.log("error-diff= %24.16e", error_inf);

    mesh.clear();
    mesh.setMeshFromQuadtree();
    app.logHead("Output mesh: %04i", 3);
    
    u_mesh_diff.name() = "u_compare";
    mesh.addMeshFunction(u_mesh_diff);
    mesh.toVTK("toybox-comparison", 0);

    return EXIT_SUCCESS;
}