/**
 * @file main.cpp : helmholtz
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Sets up and solves a Helmholtz equation
 * 
 * Solves a Helmholtz equation:
 * 
 * laplacian( u ) + lambda * u = f
 * 
 * subject to Dirichlet boundary conditions provided by the exact solution.
 * 
 * Due to the highly oscillatory nature of this problem, there is no adaptive
 * refinement. This problem is solved on a uniformly refined mesh.
 * 
 * Usage:
 * mpirun -n <mpi-ranks> ./helmholtz <min-level> <max-level> <nx> <ny>
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
 * @brief Main driver for helmholtz
 * 
 * Solves a Helmholtz equation with EllipticForest
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
        return 1.0;
    };
    solver.lambda_function = [&](double x, double y){
        return lambda;
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
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
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
        //      "helmholtz-mesh.pvtu"            : Parallel header file for mesh and data
        //      "helmholtz-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("helmholtz");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}
