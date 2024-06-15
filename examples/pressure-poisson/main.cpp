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

static double PRESSURE_INLET = 100.;
static double PRESSURE_OUTLET = 0.;
static double RHO1 = 100.0;
static double RHO2 = 1.0;

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
    if (x > 0.5 && y < 0.5) {
        return RHO1;
    }
    else {
        return RHO2;
    }
    return 0.;
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

    bool homogeneous_rhs = true;
    app.options.setOption("homogeneous-rhs", homogeneous_rhs);

    bool homogeneous_beta = false;
    app.options.setOption("homogeneous-beta", homogeneous_beta);

    bool vtk_flag = true;
    app.options.setOption("vtk-flag", vtk_flag);
    
    int min_level = 2;
    app.options.setOption("min-level", min_level);
    
    int max_level = 6;
    app.options.setOption("max-level", max_level);

    double x_lower = 0.;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 1.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = 0.;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 1.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 16;
    app.options.setOption("nx", nx);
    
    int ny = 16;
    app.options.setOption("ny", ny);

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
    EllipticForest::FiniteVolumeNodeFactory node_factory(MPI_COMM_WORLD, solver);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            return (((0.45 < x) && (x < 0.55)) || ((0.45 < y) && (y < 0.55))) && ((x > 0.5) && (y < 0.5));
        },
        0,
        min_level,
        max_level,
        root_patch,
        node_factory
    );

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

    // 4. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        if (side == 0) {
            *a = 1.0;
            *b = 0.0;
            return PRESSURE_INLET;
        }
        else if (side == 1) {
            *a = 1.0;
            *b = 0.0;
            return PRESSURE_OUTLET;
        }
        else if (side == 2 || side == 3) {
            *a = 0.0;
            *b = 1.0;
            return 0.;
        }
    });

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
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.alpha_function(x, y);
            },
            "alpha_fn"
        );
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.beta_function(x, y);
            },
            "beta_fn"
        );
        mesh.addMeshFunction(
            [&](double x, double y){
                return solver.lambda_function(x, y);
            },
            "lambda_fn"
        );

        // Write VTK files:
        //      "pressure-poisson-mesh.pvtu"            : Parallel header file for mesh and data
        //      "pressure-poisson-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("pressure-poisson");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}