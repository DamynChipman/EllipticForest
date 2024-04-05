/**
 * @file main.cpp : elliptic-single
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Sets up and solves an elliptic PDE using the Hierarchical Poincaré-Steklov (HPS) method.
 * 
 * Solves Poisson's equation:
 * 
 * laplacian( u ) = f
 * 
 * subject to Dirichlet boundary conditions provided by the exact solution.
 * 
 * By default, this is set to solve for the exact solution:
 * 
 * u(x,y) = sin(x) + sin(y)
 * 
 * thus,
 * 
 * f(x,y) = -sin(x) - sin(y) = -u(x,y).
 * 
 * EllipticForest solves this by creating a mesh and refining it according to the curvature of the
 * solution (i.e., the right-hand side function `f`). The build, upwards, and solve stages are used
 * to do the factorization and application of the solution operators. The solution is output to VTK
 * files to be viewed with your favorite visualization tool (VisIt is mine!)
 * 
 */

#include <cmath>
#include <iostream>
#include <string>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

/**
 * @brief Wrapper of cmath Bessel Y function
 * 
 * @param n Order of Bessel function
 * @param x Value of x
 * @return double 
 */
double besselY(int n, double x) {
    return yn(n, x);
}

/**
 * @brief Main driver for elliptic-single
 * 
 * Solves an elliptic PDE with user defined setup functions using EllipticForest
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
    
    double threshold = 1.2;
    app.options.setOption("refinement-threshold", threshold);
    
    int n_solves = 1;
    app.options.setOption("n-solves", n_solves);
    
    double x_lower = -10.0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 10.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = -10.0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 10.0;
    app.options.setOption("y-upper", y_upper);
    
    int min_level = 0;
    int max_level = 5;
    int nx = 16;
    int ny = 16;
    if (argc > 1) {
        min_level = atoi(argv[1]);
        max_level = atoi(argv[2]);
        nx = atoi(argv[3]);
        ny = atoi(argv[4]);
    }
    app.options.setOption("max-level", max_level);
    app.options.setOption("min-level", min_level);
    app.options.setOption("nx", nx);
    app.options.setOption("ny", ny);

    double lambda = 100.;
    double kappa = sqrt(lambda);
    double x0 = -20.;
    double y0 = 0.;
    double amplitude = 1.;
    double sigma_x = 0.1;
    double sigma_y = 0.4;

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // ====================================================
    // Create node factory and mesh
    // ====================================================
    EllipticForest::FiniteVolumeNodeFactory node_factory(MPI_COMM_WORLD);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            return true;
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        node_factory
    );

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver{};
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FivePointStencil;
    solver.alpha_function = [&](double x, double y){
        return 1.0;
    };
    solver.beta_function = [&](double x, double y){
        // bool region1 = -10. < x && x < -9.5 && y > 1.5;
        // bool region2 = -10. < x && x < -9.5 && -0.5 < y && y < 0.5;
        // bool region3 = -10. < x && x < -9.5 && y < -1.5;
        // if (region1 || region2 || region3) {
        //     return 100.;
        // }
        // else {
        //     return 1.;
        // }
        // if (-2. < x && x < 2. && -2. < y && y < 2.) {
        //     return 100.;
        // }
        // else {
        //     return 1.;
        // }
        return 1.0;
    };
    solver.lambda_function = [&](double x, double y){
        // if (-2. < x && x < 2. && -2. < y && y < 2.) {
        //     return 100.;
        // }
        // else {
        //     return 1.;
        // }
        return 100.0;
    };

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
    for (auto n = 0; n < n_solves; n++) {

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](double x, double y){
            return 0.;
            // return amplitude*exp(pow(x0 - x, 2)/sigma_x + pow(y0 - y, 2)/sigma_y);
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            // if (side == 0) {
            //     int n_sources = 9;
            //     double spacing = (y_upper - y_lower) / n_sources;
            //     std::vector<double> x0s(n_sources, -15.);
            //     std::vector<double> y0s(n_sources, 0.);
            //     for (auto j = 0; j < n_sources; j++) {
            //         y0s[j] = y_lower + (j + 0.5)*spacing;
            //     }

            //     double total = 0.;
            //     for (auto i = 0; i < n_sources; i++) {
            //         double r = sqrt(pow(x0s[i] - x, 2) + pow(y0s[i] - y, 2));
            //         total += besselY(0, kappa*r);
            //     }
            //     return total;
            // }
            // else {
            //     return 0.;
            // }
            // return 0.;

            double r = sqrt(pow(x0 - x, 2) + pow(y0 - y, 2));
            return besselY(0, kappa*r);
            
        });
    }

    // ====================================================
    // Write solution and functions to file
    // ====================================================
    if (vtk_flag) {
        
        // Extract out solution and right-hand side data stored on leaves
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

        // Add mesh functions to mesh
        mesh.addMeshFunction(uMesh);

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}
