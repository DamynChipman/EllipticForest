#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>

#include <SpecialMatrices.hpp>
#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

using namespace EllipticForest;

using FiniteVolumeHPS = HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double>;

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
    // return sin(x)*sinh(y);
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
    // return 0;
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
    //      "refine-coarsen-mesh-{n}.pvtu"            : Parallel header file for mesh and data
    //      "refine-coarsen-quadtree-{n}.pvtu"        : p4est quadtree structure
    mesh.toVTK("refine-coarsen", n_output);
}

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_SELF);

    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);
    
    int min_level = 1;
    app.options.setOption("min-level", min_level);
    
    int max_level = 1;
    app.options.setOption("max-level", max_level);

    double x_lower = -1.0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 1.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = -1.0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 1.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 8;
    app.options.setOption("nx", nx);
    
    int ny = 8;
    app.options.setOption("ny", ny);

    double threshold = 1.2;
    app.options.setOption("threshold", threshold);

    int level_start = 1;
    int level_finish = 5;

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
            if (node->level >= level_start) {
                return (int) false;
            }
            if (node->level <= level_start) {
                return (int) true;
            }
            return (int) false;
        }
    );
    quadtree.balance(EllipticForest::BalancePolicy::CORNER);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh(quadtree);

    FiniteVolumeHPS HPS(mpi.getComm(), mesh, solver);
    
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

    int n_output = 0;
    writeMesh(mesh, n_output++);

    // Open a file for writing a CSV
    std::ofstream output_file("error.csv");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open error file." << std::endl;
        return 1;
    }

    // Write headers
    output_file << "n,l2,lI" << std::endl;
    output_file << 1 << ",";
    output_file << std::scientific << std::setprecision(16) << error_l2 << ",";
    output_file << std::scientific << std::setprecision(16) << error_inf << std::endl;;

    for (int n = level_start+1; n <= level_finish; n++) {
        if (atoi(argv[1])) {
            mesh.quadtree.refine(
                true,
                [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                    if (node->level >= n) {
                        return (int) false;
                    }
                    if (node->level <= n) {
                        return (int) true;
                    }
                    return (int) false;
                },
                nullptr
            );
            HPS.buildStage();
        }
        else {
            mesh.quadtree.refine(
                true,
                [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                    if (node->level >= n) {
                        return (int) false;
                    }
                    if (node->level <= n) {
                        return (int) true;
                    }
                    return (int) false;
                },
                [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* parent_node, std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> children_nodes){
                    auto& tau = parent_node->data;
                    auto& alpha = children_nodes[0]->data;
                    auto& beta = children_nodes[1]->data;
                    auto& gamma = children_nodes[2]->data;
                    auto& omega = children_nodes[3]->data;
                    FiniteVolumeHPS::merge4to1(tau, alpha, beta, gamma, omega, solver);
                    return 1;
                }
            );
        }

        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });

        writeMesh(mesh, n_output++);

        double error_inf = computeMaxErrorFromExact(mesh.quadtree);
        double error_l2 = computeL2ErrorFromExact(mesh.quadtree);
        app.log("n = %04i, error-l2 = %24.16e, error-inf = %24.16e", n, error_l2, error_inf);

        output_file << n << ",";
        output_file << std::scientific << std::setprecision(16) << error_l2 << ",";
        output_file << std::scientific << std::setprecision(16) << error_inf << std::endl;;
    }

    for (int n = level_finish-1; n >= level_start; n--) {
        mesh.quadtree.coarsen(true,
            [&](std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> nodes){
                for (auto* node : nodes) {
                    if (node->level >= n+1) {
                        return (int) true;
                    }
                    if (node->level <= n+1) {
                        return (int) false;
                    }
                    return (int) false;
                }
            }
        );

        if (atoi(argv[1]))
            HPS.buildStage();
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });
        
        writeMesh(mesh, n_output++);

        double error_inf = computeMaxErrorFromExact(mesh.quadtree);
        double error_l2 = computeL2ErrorFromExact(mesh.quadtree);
        app.log("n = %04i, error-l2 = %24.16e, error-inf = %24.16e", n, error_l2, error_inf);

        output_file << n << ",";
        output_file << std::scientific << std::setprecision(16) << error_l2 << ",";
        output_file << std::scientific << std::setprecision(16) << error_inf << std::endl;;
    }

    return EXIT_SUCCESS;
}