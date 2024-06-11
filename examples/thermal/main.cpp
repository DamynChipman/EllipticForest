/**
 * @file main.cpp : thermal
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Demonstrates the ability to solve variable coefficient heat equation
 * 
 * This example solves the variable coefficient heat equation:
 * 
 * du/dt = div( beta * grad(u)) + q_volume / k
 * 
 * where:
 * 
 * u = u(x,y,t) = Temperature
 * beta = beta(x,y) = Thermal diffusivity
 * q_volume = q_volume(x,y,t) = Volumetric heat source/sink
 * k = Thermal conductivity
 * 
 * subject to the following boundary conditions:
 * 
 * u = g(x,y,t) , x,y in Omega_D (Dirichlet BC)
 * du/dn = v(x,y,t), x,y in Omega_N (Neumann BC)
 * 
 * with zero initial conditions.
 * 
 *              du/dn = 0
 *          ________________
 *          |               |
 *          |               |
 *    u=T_L |               | u=T_R
 *          |               |
 *          |               |
 *          |_______________|
 *              du/dn = 0
 * 
 * This is solved via backward Euler to obtain the following implicit scheme:
 * 
 * div( beta *  grad(u_{n+1})) - lambda * u_{n+1} = -lambda * u_{n} - q_volume / k
 * 
 * where:
 * 
 * lambda = 1 / dt
 * 
 * which is solved using EllipticForest. The mesh is built with according to a user defined
 * criteria (defaults to refinement at the edges of the domain). The set of solution operators
 * are built and then used for each time step, resulting in a very fast time per time step.
 * 
 * The user can mess around with most pieces of this equation to your heart's desire! Boundary
 * conditions are imposed in the functions `uWest`, `uEast`, `dudnSouth`, and `dudnNorth`. The
 * volumetric source/sinks can be added in the function `sources`. The variable thermal diffusivity
 * can be changed in `betaFunction`.
 * 
 */

#include <cmath>
#include <iostream>
#include <string>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

const double TIME_START = 0;
const double TIME_FINAL = 100;

using FiniteVolumeHPS = EllipticForest::HPSAlgorithm<EllipticForest::FiniteVolumeGrid, EllipticForest::FiniteVolumeSolver, EllipticForest::FiniteVolumePatch, double>;

/**
 * @brief Function for west boundary
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param t Time
 * @return double 
 */
double uWest(double x, double y, double t) {
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    double T_L = std::get<double>(app.options["T-left"]);
    return T_L;
}

/**
 * @brief Function for east boundary
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param t Time
 * @return double 
 */
double uEast(double x, double y, double t) {
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    double T_R = std::get<double>(app.options["T-right"]);
    return T_R;
}

/**
 * @brief Function for south boundary
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param t Time
 * @return double 
 */
double dudnSouth(double x, double y, double t) {
    return 0;
}

/**
 * @brief Function for north boundary
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param t Time
 * @return double 
 */
double dudnNorth(double x, double y, double t) {
    return 0;
}

/**
 * @brief Volumetric heat source/sink
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param t Time
 * @return double 
 */
double sources(double x, double y, double t) {
    double period = 2.0*M_PI / TIME_FINAL;
    double x0 = 5.0*sin(period*t);
    double y0 = 5.0*cos(period*t);
    double amplitude = 100.0;
    double sigma_x = 0.75;
    double sigma_y = 1.25;
    double k = 1;
    double Q = amplitude*exp(-(pow(x - x0, 2)/(2.0*pow(sigma_x,2)) + pow(y - y0, 2)/(2.0*pow(sigma_y, 2))));
    return Q / k;
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
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    double dt = std::get<double>(app.options["dt"]);
    return -1.0/dt;
}

void writeMesh(EllipticForest::Mesh<EllipticForest::FiniteVolumePatch>& mesh, double time, int n_output) {
    auto& app = EllipticForest::EllipticForestApp::getInstance();
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
    mesh.addMeshFunction(
        [&](double x, double y){
            return sources(x, y, time);
        },
        "sources"
    );

    // Write VTK files:
    //      "thermal-mesh-{n}.pvtu"            : Parallel header file for mesh and data
    //      "thermal-quadtree-{n}.pvtu"        : p4est quadtree structure
    mesh.toVTK("thermal", n_output);
}

/**
 * @brief Main driver for thermal
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
    
    int max_level = 4;
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

    double t_start = TIME_START;
    app.options.setOption("t-start", t_start);

    double t_end = TIME_FINAL;
    app.options.setOption("t-end", t_end);

    double nt = 2000;
    app.options.setOption("nt", nt);

    double dt = (t_end - t_start) / (nt);
    app.options.setOption("dt", dt);

    int n_vtk = 2;
    app.options.setOption("n-vtk", n_vtk);

    double T_L = 20;
    app.options.setOption("T-left", T_L);

    double T_R = 20;
    app.options.setOption("T-right", T_R);

    double refine_threshold = 2.0;
    app.options.setOption("refine-threshold", refine_threshold);

    double coarsen_threshold = 1.0;
    app.options.setOption("coarsen-threshold", coarsen_threshold);

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver{};
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FivePointStencil;
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
            if (node->level < max_level) {
                return (int) true;
            }
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
                u[index] = 0.;
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

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    int n_output = 0;
    int n_debug = 0;
    for (auto n = 0; n <= nt; n++) {
        double time = t_start + n*dt;
        app.logHead("============================");
        app.logHead("n = %4i, time = %f", n, time);
        writeMesh(mesh, time, n_output++);

        if (n == 0 || atoi(argv[1])) {
            // Compute initial factorization and output initial conditions
            HPS.buildStage();
        }

        // Call the upwards stage; provide a callback to set load data on leaf patches
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
                    f[index] = -(1.0/dt)*u[index] - sources(x, y, time-dt);
                }
            }
        });

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorU(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("U: post-upwards");
        // plt::show();

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorF(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("F: post-upwards");
        // plt::show();

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            switch (side) {
                case 0:
                    // West : Dirichlet
                    // *a = 1.0;
                    // *b = 0.0;
                    // return uWest(x, y, time);
                    *a = 0.0;
                    *b = 1.0;
                    return (double) 0;

                case 1:
                    // East : Dirichlet
                    // *a = 1.0;
                    // *b = 0.0;
                    // return uEast(x, y, time);
                    *a = 0.0;
                    *b = 1.0;
                    return (double) 0;

                case 2:
                    // South : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return dudnSouth(x, y, time);

                case 3:
                    // North : Neumann
                    *a = 0.0;
                    *b = 1.0;
                    return dudnNorth(x, y, time);
                
                default:
                    break;
            }

            return 0.0;
        });

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorU(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("U: post-solve");
        // plt::show();

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorF(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("F: post-solve");
        // plt::show();

        writeMesh(mesh, time, n_output++);

        // ====================================================
        // Write solution and functions to file
        // ====================================================
        // if (vtk_flag && n % n_vtk == 0) {
        // }

        // ====================================================
        // Refine and coarsen the mesh
        // ====================================================
        mesh.quadtree.adapt(
            min_level,
            max_level,
            // Coarsen function
            [&](std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> nodes){
                bool coarsen_nodes = true;
                for (auto* node : nodes) {
                    if (node->leaf) {
                        if (node->level <= min_level) {
                            return (int) false;
                        }

                        auto& patch = node->data;
                        auto& grid = patch.grid();
                        auto& u = patch.vectorU();
                        auto nx = grid.nx();
                        auto ny = grid.ny();
                        for (int i = 0; i < nx; i++) {
                            for (int j = 0; j < ny; j++) {
                                double x = grid(0, i);
                                double y = grid(1, j);
                                if (sources(x, y, time) > coarsen_threshold) {
                                    coarsen_nodes = false;
                                }
                            }
                        }
                        if (coarsen_nodes == false) {
                            return (int) false;
                        }
                    }
                }
                return (int) coarsen_nodes;
            },
            // Refine function
            [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
                bool refine_node = false;
                if (node->leaf) {
                    if (node->level >= max_level) {
                        return (int) false;
                    }

                    auto& patch = node->data;
                    auto& grid = patch.grid();
                    auto& u = patch.vectorU();
                    auto nx = grid.nx();
                    auto ny = grid.ny();
                    for (int i = 0; i < nx; i++) {
                        for (int j = 0; j < ny; j++) {
                            double x = grid(0, i);
                            double y = grid(1, j);
                            refine_node = (sources(x, y, time) > refine_threshold);
                            if (refine_node) {
                                return (int) refine_node;
                            }
                        }
                    }
                }
                return (int) refine_node;
            },
            // Propagate function
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

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorU(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("U: post-adapt");
        // plt::show();

        // mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //     if (node->leaf) {
        //         auto& patch = node->data;
        //         auto& grid = patch.grid();
        //         plt::scatter3(grid, patch.vectorF(), 1.0, {{"c", "r"}}, 1);
        //     }
        //     return 1;
        // });
        // plt::title("F: post-adapt");
        // plt::show();
        
        // mesh.quadtree.merge(
        //     [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* leaf_node) {
        //         return 1;
        //     },
        //     [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* parent_node, std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> child_nodes) {
        //         EllipticForest::FiniteVolumePatch& tau = parent_node->data;
        //         EllipticForest::FiniteVolumePatch& alpha = child_nodes[0]->data;
        //         EllipticForest::FiniteVolumePatch& beta = child_nodes[1]->data;
        //         EllipticForest::FiniteVolumePatch& gamma = child_nodes[2]->data;
        //         EllipticForest::FiniteVolumePatch& omega = child_nodes[3]->data;
        //         std::vector<std::size_t> sizes = {
        //             alpha.grid().nx(),
        //             beta.grid().nx(),
        //             gamma.grid().nx(),
        //             omega.grid().nx()
        //         };
        //         int min_size = *std::min_element(sizes.begin(), sizes.end());
        //         EllipticForest::FiniteVolumeGrid merged_grid(MPI_COMM_SELF, 2*min_size, alpha.grid().xLower(), beta.grid().xUpper(), 2*min_size, alpha.grid().yLower(), gamma.grid().yUpper());
        //         tau.grid() = merged_grid;
        //         return 1;
        //     }
        // );
        // writeMesh(mesh, time, n_output++);

        // for (int l = max_level; l > min_level; l--) {
        //     mesh.quadtree.coarsen(false,
        //         [&](std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> nodes){
        //             bool coarsen_nodes = false;
        //             for (auto* child_node : nodes) {
        //                 if (child_node->leaf) {
        //                     auto& patch = child_node->data;
        //                     auto& grid = patch.grid();
        //                     auto& u = patch.vectorU();
        //                     auto nx = grid.nx();
        //                     auto ny = grid.ny();

        //                     if (child_node->level <= min_level) {
        //                         return 0;
        //                     }

        //                     for (int i = 0; i < nx; i++) {
        //                         for (int j = 0; j < ny; j++) {
        //                             double x = grid(0, i);
        //                             double y = grid(1, j);
        //                             // app.log("COARSEN: source = " + std::to_string(sources(x, y, time)));
        //                             coarsen_nodes = (sources(x, y, time) < coarsen_threshold);

        //                             if (coarsen_nodes) {
        //                                 break;
        //                             }
        //                         }
        //                         if (coarsen_nodes) {
        //                             break;
        //                         }
        //                     }
        //                 }
        //             }
        //             return (int) coarsen_nodes;
        //         }
        //     );
        //     mesh.quadtree.balance(EllipticForest::BalancePolicy::FACE);
        // }
        // for (int l = min_level; l < max_level; l++) {
        //     mesh.quadtree.refine(false,
        //         [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        //             bool refine_node = false;
        //             if (node->leaf) {
        //                 auto& patch = node->data;
        //                 auto& grid = patch.grid();
        //                 auto& u = patch.vectorU();
        //                 auto nx = grid.nx();
        //                 auto ny = grid.ny();

        //                 if (node->level >= max_level) {
        //                     return 0;
        //                 }

        //                 for (int i = 0; i < nx; i++) {
        //                     for (int j = 0; j < ny; j++) {
        //                         double x = grid(0, i);
        //                         double y = grid(1, j);
        //                         refine_node = (sources(x, y, time) > refine_threshold);

        //                         // int I_ij = j + i*ny;
        //                         // int I_ip1j = j + (i+1)*ny;
        //                         // int I_im1j = j + (i-1)*ny;
        //                         // int I_ijp1 = (j+1) + i*ny;
        //                         // int I_ijm1 = (j-1) + i*ny;

        //                         // double u_ij = u(I_ij);
        //                         // double u_ip1j = u(I_ip1j);
        //                         // double u_im1j = u(I_im1j);
        //                         // double u_ijp1 = u(I_ijp1);
        //                         // double u_ijm1 = u(I_ijm1);

        //                         // double dudx = (u_ip1j - u_im1j) / (2.0*grid.dx());
        //                         // double dudy = (u_ijp1 - u_ijm1) / (2.0*grid.dy());
        //                         // double mag_grad_u = sqrt(pow(dudx, 2) + pow(dudy, 2));
        //                         // refine_node = mag_grad_u > refine_threshold;

        //                         if (refine_node) {
        //                             break;
        //                         }
        //                     }
        //                     if (refine_node) {
        //                         break;
        //                     }
        //                 }
        //             }
        //             return (int) refine_node;
        //         }
        //     );
        //     mesh.quadtree.balance(EllipticForest::BalancePolicy::FACE);
        // }
        // mesh.quadtree.merge(
        //     [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* patch){
        //         return 1;
        //     },
        //     [&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* parent_node, std::vector<EllipticForest::Node<EllipticForest::FiniteVolumePatch>*> children_nodes){
        //         EllipticForest::FiniteVolumeHPS::coarsen(parent_node->data, children_nodes[0]->data, children_nodes[1]->data, children_nodes[2]->data, children_nodes[3]->data);
        //         return 1;
        //     }
        // );
    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}