/**
 * @file main.cpp : weak-scaling
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
#include <algorithm>

#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

class PolarStarPoissonProblem {

public:

    int nPolar;
    std::vector<double> x0s;
    std::vector<double> y0s;
    std::vector<double> r0s;
    std::vector<double> r1s;
    std::vector<double> ns;
    double eps_disk = 0.015625;

    PolarStarPoissonProblem() {
        nPolar = 1;
        x0s = {0};
        y0s = {0};
        r0s = {0.4};
        r1s = {0.45};
        ns = {4};
    }

    PolarStarPoissonProblem(int nPolar, std::vector<double> x0s, std::vector<double> y0s, std::vector<double> r0s, std::vector<double> r1s, std::vector<double> ns, double epsilon) :
        nPolar(nPolar),
        x0s(x0s),
        y0s(y0s),
        r0s(r0s),
        r1s(r1s),
        ns(ns),
        eps_disk(epsilon)
            {}

    std::string name() { return "polar-star-poisson"; }

    double lambda() { return 0.0; }

    double u(double x, double y) {
        double res = 0;
        for (auto i = 0; i < nPolar; i++) {
            double x0 = x0s[i];
            double y0 = y0s[i];
            double r = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
            double theta = atan2(y - y0, x - x0);
            res += 1.0 - hsmooth(i, r, theta);
        }
        return res;
    }

    double f(double x, double y) {
        double res = 0;
        for (auto i = 0; i < nPolar; i++) {
            double x0 = x0s[i];
            double y0 = y0s[i];
            double r = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
            double theta = atan2(y - y0, x - x0);
            res -= hsmooth_laplacian(i, r, theta);
        }
        return res;
    }

    double dudx(double x, double y) {
        return 0;
    }

    double dudy(double x, double y) {
        return 0;
    }

private:

    double sech(double x) {
        return 1.0 / cosh(x);
    }

    void polar_interface_complete(int ID, double theta, double& p, double& dpdtheta, double& d2pdtheta2) {
        double r0 = r0s[ID];
        double r1 = r1s[ID];
        int n = ns[ID];

        p = r0*(1.0 + r1*cos(n*theta));
        dpdtheta = r0*(-n*r1*sin(n*theta));
        d2pdtheta2 = r0*(-pow(n,2)*r1*cos(n*theta));
    }

    double polar_interface(int ID, double theta) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);
        return p;
    }

    double hsmooth(int ID, double r, double theta) {
        double p = polar_interface(ID, theta);
        return (tanh((r - p)/eps_disk) + 1.0) / 2.0;
    }

    void hsmooth_grad(int ID, double r, double theta, double& grad_x, double& grad_y) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);
        
        double eps_disk2 = pow(eps_disk, 2);
        double sech2 = pow(sech((r - p)/eps_disk), 2);
        grad_x = sech2 / eps_disk2;
        grad_y = -dpdtheta*sech2/(eps_disk2*r);

    }

    double hsmooth_laplacian(int ID, double r, double theta) {
        double p = 0;
        double dpdtheta = 0;
        double d2pdtheta2 = 0;
        polar_interface_complete(ID, theta, p, dpdtheta, d2pdtheta2);

        double eps_disk2 = pow(eps_disk, 2);
        double sech2 = pow(sech((r-p)/eps_disk), 2);
        double t = tanh((r-p)/eps_disk);
        double st = t*sech2;
        double s1 = pow(dpdtheta,2)*st/eps_disk2;
        double s2 = d2pdtheta2*sech2/(2*eps_disk);
        double s3 = st/eps_disk2;
        double s4 = sech2/(2*eps_disk*r);
        return (-s1-s2)/pow(r, 2) - s3 + s4;
    }

};

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

    std::vector<int> allowed_ranks = {1, 4, 16, 64, 256, 1024};
    if (std::find(allowed_ranks.begin(), allowed_ranks.end(), mpi.getSize()) == allowed_ranks.end()) {
        app.logHead("Invalid number of ranks - only run with {1, 4, 16, 64, 256, 1024}");
        return EXIT_FAILURE;
    }

    double extent = sqrt((double) mpi.getSize());

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
    
    int n_solves = 1;
    app.options.setOption("n-solves", n_solves);
    
    double x_lower = 0;
    app.options.setOption("x-lower", x_lower);

    double x_upper = extent;
    app.options.setOption("x-upper", x_upper);

    double y_lower = 0;
    app.options.setOption("y-lower", y_lower);

    double y_upper = extent;
    app.options.setOption("y-upper", y_upper);
    
    int min_level = 0;
    app.options.setOption("min-level", min_level);
    
    int max_level = 6 + log2((int) extent);
    app.options.setOption("max-level", max_level);
    
    int nx = 16;
    app.options.setOption("nx", nx);
    
    int ny = 16;
    app.options.setOption("ny", ny);

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    int n_polar = mpi.getSize();
    std::vector<double> x0s(n_polar);
    std::vector<double> y0s(n_polar);
    std::vector<double> r0s(n_polar);
    std::vector<double> r1s(n_polar);
    std::vector<double> ns(n_polar);
    double epsilon = 0.015625;
    for (int i = 0; i < (int) extent; i++) {
        for (int j = 0; j < (int) extent; j++) {
            x0s[j + i*(int)extent] = 0.5 + (double) i;
            y0s[j + i*(int)extent] = 0.5 + (double) j;
        }
    }
    for (int n = 0; n < n_polar; n++) {
        r0s[n] = 0.3;
        r1s[n] = 0.4;
        ns[n] = 4;
    }
    PolarStarPoissonProblem pde(n_polar, x0s, y0s, r0s, r1s, ns, epsilon);

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
            double f = pde.f(x, y);
            return fabs(f) > threshold;
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
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

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
            return pde.f(x, y);
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return pde.u(x, y);
        });
    }

    // Get problem metrics
    int effective_resolution = pow(2, max_level) * nx;
    int effective_dofs = pow(effective_resolution, 2);
    int actual_dofs_local = 0;
    mesh.quadtree.traversePreOrder([&](EllipticForest::Node<EllipticForest::FiniteVolumePatch>* node){
        if (node->leaf) {
            auto& patch = node->data;
            auto& grid = patch.grid();
            actual_dofs_local += pow(grid.nx(), 2);
        }
        return 1;
    });
    int actual_dofs_global = 0;
    EllipticForest::MPI::reduce(actual_dofs_local, actual_dofs_global, MPI_SUM, 0, MPI_COMM_WORLD);
    app.logHead("eff_resolution = %16i, eff_dofs = %16i, act_dofs = %16i, dofs_per_rank = %16i", effective_resolution, effective_dofs, actual_dofs_global, actual_dofs_global / mpi.getSize());

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
                return pde.u(x, y);
            },
            "u_exact"
        );

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("weak-scaling");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}
