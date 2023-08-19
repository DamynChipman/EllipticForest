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
 * are built and then used for each time step, resulting in a very fast time per timestep.
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
#include <format>

#include <EllipticForest.hpp>

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
    return T_L + 20.*sin(t)*sin(y);
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
    return 0.0;
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

/**
 * @brief Main driver for thermal
 * 
 * Solves the variable coefficiant heat equation via implicit backward Euler
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
    
    int min_level = 1;
    app.options.setOption("min-level", min_level);
    
    int max_level = 6;
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

    double t_start = 0.0;
    app.options.setOption("t-start", t_start);

    double t_end = 100.0;
    app.options.setOption("t-end", t_end);

    double nt = 100;
    app.options.setOption("nt", nt);

    double dt = (t_end - t_start) / (nt);
    app.options.setOption("dt", dt);

    int n_vtk = 10;
    app.options.setOption("n-vtk", n_vtk);

    double T_L = 40;
    app.options.setOption("T-left", T_L);

    double T_R = 20;
    app.options.setOption("T-right", T_R);

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::Petsc::PetscGrid grid(nx, ny, x_lower, x_upper, y_lower, y_upper);
    EllipticForest::Petsc::PetscPatch root_patch(grid);

    // ====================================================
    // Create node factory and mesh
    // ====================================================
    EllipticForest::Petsc::PetscPatchNodeFactory nodeFactory{};
    EllipticForest::Mesh<EllipticForest::Petsc::PetscPatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            double eps = 1.0;
            return (x_lower + eps > x || x > x_upper - eps);
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        nodeFactory
    );

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::Petsc::PetscPatchSolver solver{};
    solver.setAlphaFunction(alphaFunction);
    solver.setBetaFunction(betaFunction);
    solver.setLambdaFunction(lambdaFunction);

    // ====================================================
    // Create and run HPS solver
    // ====================================================
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::Petsc::PetscGrid,
         EllipticForest::Petsc::PetscPatchSolver,
         EllipticForest::Petsc::PetscPatch,
         double>
            HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    int n_output = 0;
    for (auto n = 0; n <= nt; n++) {

        double time = t_start + n*dt;
        app.logHead("============================");
        app.logHead("n = %4i, time = %f", n, time);

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](EllipticForest::Petsc::PetscPatch& patch){
            auto& grid = patch.grid();
            int nx = grid.nPointsX();
            int ny = grid.nPointsY();
            patch.vectorF() = EllipticForest::Vector<double>(nx*ny);
            auto& u = patch.vectorU();
            auto& f = patch.vectorF();
            if (n == 0) {
                // Set initial condition to RHS function
                for (auto i = 0; i < nx; i++) {
                    for (auto j = 0; j < ny; j++) {
                        int index = j + i*ny;
                        f[index] = 0.0;
                    }
                }
            }
            else {
                // Update RHS function with previous time step
                for (auto i = 0; i < nx; i++) {
                    for (auto j = 0; j < ny; j++) {
                        int index = j + i*ny;
                        double x = grid(0, i);
                        double y = grid(1, j);
                        f[index] = -(1.0/dt)*u[index] - sources(x, y, time);
                    }
                }
            }
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            switch (side) {
                case 0:
                    // West : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return uWest(x, y, time);

                case 1:
                    // East : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return uEast(x, y, time);

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


        // ====================================================
        // Write solution and functions to file
        // ====================================================
        if (vtk_flag && n % n_vtk == 0) {
            app.logHead("Output mesh: %04i", n_output);

            mesh.clearMeshFunctions();
            
            // Extract out solution and right-hand side data stored on leaves
            EllipticForest::Vector<double> uMesh{};
            uMesh.name() = "u_soln";
            
            EllipticForest::Vector<double> fMesh{};
            fMesh.name() = "f_rhs";
            
            mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::Petsc::PetscPatch>* node){
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
                    return solver.alphaFunction(x, y);
                },
                "alpha_fn"
            );
            mesh.addMeshFunction(
                [&](double x, double y){
                    return solver.betaFunction(x, y);
                },
                "beta_fn"
            );
            mesh.addMeshFunction(
                [&](double x, double y){
                    return solver.lambdaFunction(x, y);
                },
                "lambda_fn"
            );

            // Write VTK files:
            //      "thermal-mesh-{n}.pvtu"            : Parallel header file for mesh and data
            //      "thermal-quadtree-{n}.pvtu"        : p4est quadtree structure
            mesh.toVTK("thermal", n_output);
            n_output++;
        }

        app.logHead("============================");
    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}