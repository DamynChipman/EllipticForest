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

static double A_COEF = 10.0;
static double B_COEF = 9.0;
static double C_COEF = 10.0;

static double RHO1 = 100.0;
static double RHO2 = 1.0;

static double NU_WATER = 0.001;

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
    // return 1.0;
    // return sin(x)*sin(y) + 2.0;
    
    if (x > 0.5 && y < 0.5) {
        return RHO1;
    }
    else {
        return RHO2;
    }

    // double k = 1.;
    // double r = sqrt(pow(0.5 - x, 2) + pow(0.5 - y, 2));
    // if (r < 0.25) k = 100.;
    // return k / NU_WATER;

    // return 1.0;

    // return cos(M_PI*x)*sin(M_PI*y) + 2.;

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
    // return sin(x) + sin(y);
    // double beta = sqrt(pow(A_COEF, 2) + pow(B_COEF, 2));
    // return exp(-lambdaFunction(x,y)/C_COEF)*sin(beta*y);

    // return sin(x) + sin(y);

    // return -(cos(2.0*M_PI*x)*cos(2.0*M_PI*y)/(8.0*pow(M_PI, 2)));
    // return 0.;

    // return x*(1. - x)*y*(1. - y)*exp(x*y);

    return 0.;
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
    // return -uFunction(x, y);
    // return -B_COEF*(2.*A_COEF + C_COEF)*sin(C_COEF*x)*uFunction(x,y);
    // return pow(cos(y),2)*sin(x) + pow(cos(x),2)*sin(y) - sin(x)*(2 + sin(x)*sin(y)) - sin(y)*(2 + sin(x)*sin(y));
    // return cos(2.0*M_PI*x)*cos(2.0*M_PI*y);

    return 0.;

    // return -2*(2*(-pow(M_E,x*y) + pow(M_E,x*y)*(1 - x)*y) + x*(-2*pow(M_E,x*y)*y + pow(M_E,x*y)*(1 - x)*pow(y,2))) + 2*(1 - 2*y)*(2*(pow(M_E,x*y)*(1 - x) - pow(M_E,x*y)*x + pow(M_E,x*y)*(1 - x)*x*y) + x*(-2*pow(M_E,x*y) + 2*pow(M_E,x*y)*(1 - x)*y - 2*pow(M_E,x*y)*x*y + pow(M_E,x*y)*(1 - x)*x*pow(y,2))) + (1 - y)*y*(2*(2*pow(M_E,x*y)*(1 - x)*x - pow(M_E,x*y)*pow(x,2) + pow(M_E,x*y)*(1 - x)*pow(x,2)*y) + x*(2*pow(M_E,x*y)*(1 - x) - 4*pow(M_E,x*y)*x + 4*pow(M_E,x*y)*(1 - x)*x*y - 2*pow(M_E,x*y)*pow(x,2)*y + pow(M_E,x*y)*(1 - x)*pow(x,2)*pow(y,2)));

    // return M_PI*(exp(x*y)*(1 - x)*x*(1 - y) - exp(x*y)*(1 - x)*x*y + exp(x*y)*(1 - x)*exp(2)*(1 - y)*y)*cos(M_PI*x)*cos(M_PI*y) - M_PI*(exp(x*y)*(1 - x)*(1 - y)*y - exp(x*y)*x*(1 - y)*y + exp(x*y)*(1 - x)*x*(1 - y)*exp(2))*sin(M_PI*x)*sin(M_PI*y) + (-2*exp(x*y)*(1 - x)*x + 2*exp(x*y)*(1 - x)*exp(2)*(1 - y) - 2*exp(x*y)*(1 - x)*exp(2)*y + exp(x*y)*(1 - x)*exp(3)*(1 - y)*y)*(2 + cos(M_PI*x)*sin(M_PI*y)) + (-2*exp(x*y)*(1 - y)*y + 2*exp(x*y)*(1 - x)*(1 - y)*exp(2) - 2*exp(x*y)*x*(1 - y)*exp(2) + exp(x*y)*(1 - x)*x*(1 - y)*exp(3))*(2 + cos(M_PI*x)*sin(M_PI*y));

    // return -2*pow(M_E,x*y)*(1 - x)*x + 2*pow(M_E,x*y)*(1 - x)*pow(x,2)*(1 - y) - 2*pow(M_E,x*y)*(1 - x)*pow(x,2)*y - 2*pow(M_E,x*y)*(1 - y)*y + pow(M_E,x*y)*(1 - x)*pow(x,3)*(1 - y)*y + 2*pow(M_E,x*y)*(1 - x)*(1 - y)*pow(y,2) - 2*pow(M_E,x*y)*x*(1 - y)*pow(y,2) + pow(M_E,x*y)*(1 - x)*x*(1 - y)*pow(y,3);
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

    bool homogeneous_beta = false;
    app.options.setOption("homogeneous-beta", homogeneous_beta);

    bool vtk_flag = true;
    app.options.setOption("vtk-flag", vtk_flag);
    
    double threshold = 1.2;
    app.options.setOption("refinement-threshold", threshold);
    
    int n_solves = 1;
    app.options.setOption("n-solves", n_solves);
    
    int min_level = 2;
    app.options.setOption("min-level", min_level);
    
    int max_level = 8;
    app.options.setOption("max-level", max_level);

    double x_lower = 0.;
    app.options.setOption("x-lower", x_lower);

    double x_upper = 1.0;
    app.options.setOption("x-upper", x_upper);

    double y_lower = 0.;
    app.options.setOption("y-lower", y_lower);

    double y_upper = 1.0;
    app.options.setOption("y-upper", y_upper);
    
    int nx = 32;
    app.options.setOption("nx", nx);
    
    int ny = 32;
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
            // return true;
            return (((0.45 < x) && (x < 0.55)) || ((0.45 < y) && (y < 0.55))) && ((x > 0.5) && (y < 0.5));
        },
        threshold,
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

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    for (auto n = 0; n < n_solves; n++) {

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            // *a = 1.0;
            // *b = 0.0;
            // return uFunction(x, y);

            if (side == 0) {
                *a = 1.0;
                *b = 0.0;
                return 100.;
            }
            else if (side == 1) {
                *a = 1.0;
                *b = 0.0;
                return 0.;
            }
            else if (side == 2 || side == 3) {
                *a = 0.0;
                *b = 1.0;
                return 0.;
            }
        });
    }

    // ====================================================
    // Compute error
    // ====================================================
    double l1_error = 0;
    double l2_error = 0;
    double lI_error = 0;
    int n_leaf_patches = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();
            for (auto i = 0; i < grid.nx(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.ny(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.ny();
                    double diff = patch.vectorU()[index] - uFunction(x, y);
                    l1_error += (grid.dx()*grid.dy())*fabs(diff);
                    l2_error += (grid.dx()*grid.dy())*pow(fabs(diff), 2);
                    lI_error = fmax(lI_error, fabs(diff));
                }
            }
            n_leaf_patches++;
        }
        return 1;
    });
    double area = (x_upper - x_lower) * (y_upper - y_lower);
    l1_error = l1_error / area;
    l2_error = sqrt(l2_error / area);
    int resolution = pow(2,max_level)*nx;
    int nDOFs = n_leaf_patches * (nx * ny);

    // Compute size of quadtree and data
    double size_MB = 0;
    mesh.quadtree.traversePostOrder([&](EllipticForest::FiniteVolumePatch& patch){
        size_MB += patch.dataSize();
    });
    app.log("L1: %24.16e, L2: %24.16e, LI: %24.16e", l1_error, l2_error, lI_error);
    app.log("resolution: %16i, nDofs: %16i, memory: %16.8f [MB]", resolution, nDOFs, size_MB);

    // ====================================================
    // Write solution and functions to file
    // ====================================================
    if (vtk_flag) {
        
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
        mesh.addMeshFunction(
            [&](double x, double y){
                return uFunction(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}