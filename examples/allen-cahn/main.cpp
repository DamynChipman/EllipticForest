/**
 * @file main.cpp : allen-cahn
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Solves the Allen-Cahn equation
 * 
 * 
 */

#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

double omegaWest(double x, double y, double t) {
    return 0;
}

double omegaEast(double x, double y, double t) {
    return 0;
}

double omegaSouth(double x, double y, double t) {
    return 0;
}

double omegaNorth(double x, double y, double t) {
    return 0;
}

double uInitial(double x, double y) {
    double u = -1.;
    double xp = -1. + 2.*x;
    double yp = -1. + 2.*y;
    double rp = sqrt(pow(xp, 2) + pow(yp, 2));
    if (rp < 0.4) u = 1.;

    for (int i = 0; i < 4; i++) {
        double th = (i - 1)*M_PI/2.;
        double x0 = 0.6*cos(th);
        double y0 = 0.6*sin(th);
        rp = sqrt(pow(xp - x0, 2) + pow(yp - y0, 2));
        if (rp < 0.2) u = 1.;

        th = M_PI/4. + (i - 1)*M_PI/2.;
        x0 = 0.55*cos(th);
        y0 = 0.55*sin(th);
        rp = sqrt(pow(xp - x0, 2) + pow(yp - y0, 2));
        if (rp < 0.15) u = 1.;
    }

    return u;
}

double energyPotential(double x, double y, double t, double u) {
    auto& app = EllipticForest::EllipticForestApp::getInstance();
    double epsilon = std::get<double>(app.options["epsilon"]);
    return (1./pow(epsilon,2))*(pow(u,3) - u);
}

/**
 * @brief User-defined alpha function
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @return double 
 */
double alphaFunction(double x, double y) {
    auto& app = EllipticForest::EllipticForestApp::getInstance();
    double epsilon = std::get<double>(app.options["epsilon"]);
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
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    double dt = std::get<double>(app.options["dt"]);
    return -1.0/dt;
}

/**
 * @brief Main driver for allen-cahn
 * 
 * Solves the variable coefficient heat equation via implicit backward Euler
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv) {

    // ====================================================
    // Initialize app and MPI
    // ====================================================
    EllipticForest::EllipticForestApp app(&argc, &argv);
    EllipticForest::MPI::MPIObject mpi(MPI_COMM_WORLD);

    // ====================================================
    // Setup options
    // ====================================================
    bool cache_operators = false;
    app.options.setOption("cache-operators", cache_operators);

    bool homogeneous_rhs = false;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);

    bool vtk_flag = true;
    app.options.setOption("vtk-flag", vtk_flag);
    
    double threshold = 1.0;
    app.options.setOption("refinement-threshold", threshold);
    
    int min_level = 2;
    app.options.setOption("min-level", min_level);
    
    int max_level = 3;
    app.options.setOption("max-level", max_level);

    double x_lower = 0.0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 1.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = 0.0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 1.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 8;
    app.options.setOption("nx", nx);
    
    int ny = 8;
    app.options.setOption("ny", ny);

    double t_start = 0;
    app.options.setOption("t-start", t_start);

    double t_end = 0.001;
    app.options.setOption("t-end", t_end);

    double nt = 100;
    app.options.setOption("nt", nt);

    double dt = (t_end - t_start) / (nt);
    app.options.setOption("dt", dt);

    int n_vtk = 1;
    app.options.setOption("n-vtk", n_vtk);

    double refine_threshold = 1e-1;
    app.options.setOption("refine-threshold", refine_threshold);

    double coarsen_threshold = 5e-3;
    app.options.setOption("coarsen-threshold", coarsen_threshold);

    double epsilon = .1;
    app.options.setOption("epsilon", epsilon);

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver{};
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

    // ====================================================
    // Create node factory and mesh
    // ====================================================
    EllipticForest::FiniteVolumeNodeFactory node_factory(mpi.getComm(), solver);
    EllipticForest::Quadtree<EllipticForest::FiniteVolumePatch> quadtree(mpi.getComm(), root_patch, node_factory, {x_lower, x_upper, y_lower, y_upper});
    quadtree.refine(true,
        [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
            return (int) node->level < min_level;
        }
    );
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh(quadtree);

    // ====================================================
    // Set up initial conditions
    // ====================================================
    mesh.iteratePatches([&](EllipticForest::FiniteVolumePatch& patch){
        auto& grid = patch.grid();
        auto& u = patch.vectorU();
        auto& f = patch.vectorF();
        u = EllipticForest::Vector<double>(grid.nx()*grid.ny());
        f = EllipticForest::Vector<double>(grid.nx()*grid.ny());
        for (int i = 0; i < grid.nx(); i++) {
            for (int j = 0; j < grid.ny(); j++) {
                double x = grid(0, i);
                double y = grid(1, j);
                int index = j + i*grid.ny();
                u[index] = uInitial(x, y);
                f[index] = 0.;
            }
        }
    });

    // ====================================================
    // Create and run HPS solver
    // ====================================================
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::FiniteVolumeGrid,
         EllipticForest::FiniteVolumeSolver,
         EllipticForest::FiniteVolumePatch,
         double>
            HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    int n_output = 0;
    int n_adapt_output = 0;
    for (auto n = 0; n <= nt; n++) {

        double time = t_start + n*dt;
        app.logHead("============================");
        app.logHead("n = %4i, time = %f", n, time);

        double u_min = -1.;
        double u_max = 1.;
        mesh.iteratePatches([&](EllipticForest::FiniteVolumePatch& patch){
            auto& grid = patch.grid();
            auto& u = patch.vectorU().data();
            auto [v_min, v_max] = std::minmax_element(u.begin(), u.end());
            u_min = std::min(u_min, *v_min);
            u_max = std::max(u_max, *v_max);
            return;
        });
        app.log("u_min = %8.4e, u_max = %8.4e", u_min, u_max);

        // ====================================================
        // Write solution and functions to file
        // ====================================================
        if (vtk_flag && n % n_vtk == 0) {
            mesh.clear();
            mesh.setMeshFromQuadtree();
            app.logHead("Output mesh: %04i", n_output);
            
            // Extract out solution and right-hand side data stored on leaves
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

            // Write VTK files:
            //      "allen-cahn-mesh-{n}.pvtu"            : Parallel header file for mesh and data
            //      "allen-cahn-quadtree-{n}.pvtu"        : p4est quadtree structure
            mesh.toVTK("allen-cahn", n_output);
            n_output++;
        }

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](EllipticForest::FiniteVolumePatch& patch){
            auto& grid = patch.grid();
            int nx = grid.nx();
            int ny = grid.ny();
            patch.vectorF() = EllipticForest::Vector<double>(nx*ny);
            auto& u = patch.vectorU();
            auto& f = patch.vectorF();
            for (auto i = 0; i < nx; i++) {
                for (auto j = 0; j < ny; j++) {
                    int index = j + i*ny;
                    double x = grid(0, i);
                    double y = grid(1, j);
                    f[index] = -(1.0/dt)*u[index] - energyPotential(x, y, time, u[index]);
                }
            }
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            switch (side) {
                case 0:
                    // West : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return omegaWest(x, y, time);

                case 1:
                    // East : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return omegaEast(x, y, time);

                case 2:
                    // South : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return omegaSouth(x, y, time);

                case 3:
                    // North : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return omegaNorth(x, y, time);
                
                default:
                    break;
            }

            return 0.0;
        });

        // ====================================================
        // Refine and coarsen the mesh
        // ====================================================
        mesh.clear();
        mesh.setMeshFromQuadtree();
        app.logHead("Output pre-adapt mesh: %04i", n_adapt_output);
        mesh.toVTK("allen-cahn-pre-adapt", n_adapt_output);
        mesh.quadtree.adapt(
            min_level,
            max_level,
            // Coarsen function
            [&](std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> nodes){
                bool coarsen_nodes = false;
                return (int) coarsen_nodes;
            },
            // Refine function
            [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                bool refine_node = false;
                if (node->leaf) {
                    auto& patch = node->data;
                    auto& grid = patch.grid();
                    auto& u = patch.vectorU();
                    auto nx = grid.nx();
                    auto ny = grid.ny();

                    if (node->level >= max_level) {
                        return 0;
                    }

                    double ulow = -1.;
                    double uhi = 1.;
                    double umin = u[0];
                    double umax = u[0];
                    for (int i = 0; i < nx; i++) {
                        for (int j = 0; j < ny; j++) {
                            int index = j + i*ny;
                            umin = std::min(u[index], umin);
                            umax = std::max(u[index], umax);
                            // app.log("umin = %12.4f, umax = %12.4f, diff = %12.4f", umin, umax, umax - umin);
                            refine_node = (umax - umin > refine_threshold);

                            if (refine_node) {
                                break;
                            }
                        }
                        if (refine_node) {
                            break;
                        }
                    }
                }
                return (int) refine_node;
            }
        );
        mesh.clear();
        mesh.setMeshFromQuadtree();
        app.logHead("Output post-adapt mesh: %04i", n_adapt_output);
        mesh.toVTK("allen-cahn-post-adapt", n_adapt_output++);
    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}