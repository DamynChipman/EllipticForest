#include <iostream>
#include <iomanip>
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
    std::random_device rd;
    std::seed_seq ssq{rd()};
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(min, max);
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
    //      "random-mesh-{n}.pvtu"            : Parallel header file for mesh and data
    //      "random-quadtree-{n}.pvtu"        : p4est quadtree structure
    mesh.toVTK("random", n_output);
}

int main(int argc, char** argv) {

    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_SELF);

    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);
    
    int min_level = 5;
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

    FiniteVolumeGrid grid(mpi.getComm(), nx, x_lower, x_upper, ny, y_lower, y_upper);
    FiniteVolumePatch root_patch(mpi.getComm(), grid);
    FiniteVolumeSolver solver{};
    solver.solver_type = FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

    int n_adapts = 100;
    app.addTimer("time-per-iteration");
    app.addTimer("time-per-refine");
    app.addTimer("time-per-build");
    app.addTimer("time-per-upwards");
    app.addTimer("time-per-solve");
    std::vector<std::string> paths{};

    // Scope in order to reset after each part
    {
        FiniteVolumeNodeFactory node_factory(mpi.getComm(), solver);
        EllipticForest::Quadtree<EllipticForest::FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory, {x_lower, x_upper, y_lower, y_upper});
        quadtree.refine(true,
            [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                if (node->level >= min_level) {
                    return (int) false;
                }
                if (node->level <= min_level) {
                    return (int) true;
                }
                return (int) false;
            }
        );
        quadtree.balance(EllipticForest::BalancePolicy::CORNER);
        EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh(quadtree);

        FiniteVolumeHPS HPS(mpi.getComm(), mesh, solver);

        // Perform initial factorization and solve
        HPS.buildStage();
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });
    

        std::ofstream full_output_file("full-results.csv");
        if (full_output_file.is_open()) {
            full_output_file << "iteration,path,iteration-dt,refine-dt,build-dt,upwards-dt,solve-dt,l1-error,lI-error" << std::endl;
        } else {
            std::cerr << "Failed to open output file" << std::endl;
        }

        app.log("--=== Begin full-factorization loop ===--");
        // Full factorization and solve
        for (int n = 0; n < n_adapts; n++) {

            auto& iteration_timer = app.timers["time-per-iteration"];
            auto& refine_timer = app.timers["time-per-refine"];
            auto& build_timer = app.timers["time-per-build"];
            auto& upwards_timer = app.timers["time-per-upwards"];
            auto& solve_timer = app.timers["time-per-solve"];
            iteration_timer.restart();
            refine_timer.restart();
            build_timer.restart();
            upwards_timer.restart();
            solve_timer.restart();
            
            iteration_timer.start();
            
            // Get path to refine
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

            // Refine node
            refine_timer.start();
            mesh.quadtree.refineNode(path_to_refine, true);
            paths.push_back(path_to_refine);
            refine_timer.stop();

            // Perform full factorization and solve
            build_timer.start();
            HPS.buildStage();
            build_timer.stop();

            upwards_timer.start();
            HPS.upwardsStage([&](double x, double y){
                return fFunction(x, y);
            });
            upwards_timer.stop();

            solve_timer.start();
            HPS.solveStage([&](int side, double x, double y, double* a, double* b){
                *a = 1.0;
                *b = 0.0;
                return uFunction(x, y);
            });
            solve_timer.stop();

            // Get iteration time
            iteration_timer.stop();
            double iteration_dt = iteration_timer.time();
            double refine_dt = refine_timer.time();
            double build_dt = build_timer.time();
            double upwards_dt = upwards_timer.time();
            double solve_dt = solve_timer.time();
            
            // Compute error
            double error_inf = computeMaxErrorFromExact(mesh.quadtree);
            double error_l2 = computeL2ErrorFromExact(mesh.quadtree);
            app.log("iter-dt = %6.2f, path: %s, n = %04i, error-inf = %24.16e, error-l2 = %24.16e", iteration_dt, path_to_refine.c_str(), n, error_inf, error_l2);
            
            // Write iteration results to file
            if (full_output_file.is_open()) {
                full_output_file << n << ",";
                full_output_file << path_to_refine << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::fixed << iteration_dt << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::fixed << refine_dt << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::fixed << build_dt << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::fixed << upwards_dt << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::fixed << solve_dt << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::scientific << error_l2 << ",";
                full_output_file << std::setw(16) << std::setprecision(8) << std::scientific << error_inf << std::endl;
            } else {
                std::cerr << "Failed to write to output file" << std::endl;
            }

        }
    }

    {
        FiniteVolumeNodeFactory node_factory(mpi.getComm(), solver);
        EllipticForest::Quadtree<EllipticForest::FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory, {x_lower, x_upper, y_lower, y_upper});
        quadtree.refine(true,
            [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                if (node->level >= min_level) {
                    return (int) false;
                }
                if (node->level <= min_level) {
                    return (int) true;
                }
                return (int) false;
            }
        );
        quadtree.balance(EllipticForest::BalancePolicy::CORNER);
        EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh(quadtree);

        FiniteVolumeHPS HPS(mpi.getComm(), mesh, solver);

        // Perform initial factorization and solve
        HPS.buildStage();
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });

        std::ofstream adaptive_output_file("adaptive-results.csv");
        if (adaptive_output_file.is_open()) {
            adaptive_output_file << "iteration,path,iteration-dt,refine-dt,build-dt,upwards-dt,solve-dt,l1-error,lI-error" << std::endl;
        } else {
            std::cerr << "Failed to open output file" << std::endl;
        }

        app.log("--=== Begin full-factorization loop ===--");
        // Adaptive factorization and solve
        for (int n = 0; n < n_adapts; n++) {

            auto& iteration_timer = app.timers["time-per-iteration"];
            auto& refine_timer = app.timers["time-per-refine"];
            auto& build_timer = app.timers["time-per-build"];
            auto& upwards_timer = app.timers["time-per-upwards"];
            auto& solve_timer = app.timers["time-per-solve"];
            iteration_timer.restart();
            refine_timer.restart();
            build_timer.restart();
            upwards_timer.restart();
            solve_timer.restart();
            
            iteration_timer.start();

            // Refine node
            std::string path_to_refine = paths[n];
            refine_timer.start();
            mesh.quadtree.refineNode(path_to_refine, true);
            mesh.quadtree.propagate(
                path_to_refine,
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
            refine_timer.stop();

            // Perform partial factorization and solve
            upwards_timer.start();
            HPS.upwardsStage([&](double x, double y){
                return fFunction(x, y);
            });
            upwards_timer.stop();

            solve_timer.start();
            HPS.solveStage([&](int side, double x, double y, double* a, double* b){
                *a = 1.0;
                *b = 0.0;
                return uFunction(x, y);
            });
            solve_timer.stop();

            // Get iteration time
            iteration_timer.stop();
            double iteration_dt = iteration_timer.time();
            double refine_dt = refine_timer.time();
            double build_dt = build_timer.time();
            double upwards_dt = upwards_timer.time();
            double solve_dt = solve_timer.time();
            
            // Compute error
            double error_inf = computeMaxErrorFromExact(mesh.quadtree);
            double error_l2 = computeL2ErrorFromExact(mesh.quadtree);
            app.log("iter-dt = %6.2f, path: %s, n = %04i, error-inf = %24.16e, error-l2 = %24.16e", iteration_dt, path_to_refine.c_str(), n, error_inf, error_l2);
            
            // Write iteration results to file
            if (adaptive_output_file.is_open()) {
                adaptive_output_file << n << ",";
                adaptive_output_file << path_to_refine << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::fixed << iteration_dt << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::fixed << refine_dt << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::fixed << build_dt << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::fixed << upwards_dt << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::fixed << solve_dt << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::scientific << error_l2 << ",";
                adaptive_output_file << std::setw(16) << std::setprecision(8) << std::scientific << error_inf << std::endl;
            } else {
                std::cerr << "Failed to write to output file" << std::endl;
            }

        }
    }

    return EXIT_SUCCESS;
}