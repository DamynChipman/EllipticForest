/**
 * @file main.cpp : poisson
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Sets up and solves Poisson's equation using the Hierarchical Poincaré-Steklov (HPS) method
 * 
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include <fstream>

#include <EllipticForest.hpp>
#include <MPI.hpp>
#include <PETSc.hpp>
#include <Mesh.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

using PlotPair = std::pair<std::vector<int>, std::vector<double>>;

/**
 * @brief Convert a number to a string using a format specifier
 * 
 * @param x The number
 * @param format Format specifier for number output
 * @return std::string The number as a string
 */
std::string number2string(double x, std::string format="%.4f") {
    char buffer[32];
    sprintf(buffer, format.c_str(), x);
    return std::string(buffer);
}

/**
 * @brief A data structure to hold the results of the program
 */
struct ResultsData {

    std::string mode = "uniform";
    int min_level = 0;
    int max_level = 0;
    int nx = 0;
    int ny = 0;
    int effective_resolution = 0;
    int nDOFs = 0;
    double lI_error = 0;
    double l1_error = 0;
    double l2_error = 0;
    double build_time = 0;
    double upwards_time = 0;
    double solve_time = 0;
    double size_MB = 0;

    std::string csv() {
        std::string res = "";
        res += mode + ",";
        res += std::to_string(min_level) + ",";
        res += std::to_string(max_level) + ",";
        res += std::to_string(nx) + ",";
        res += std::to_string(ny) + ",";
        res += std::to_string(effective_resolution) + ",";
        res += std::to_string(nDOFs) + ",";
        res += number2string(lI_error, "%.16e") + ",";
        res += number2string(l1_error, "%.16e") + ",";
        res += number2string(l2_error, "%.16e") + ",";
        res += number2string(build_time, "%.16e") + ",";
        res += number2string(upwards_time, "%.16e") + ",";
        res += number2string(solve_time, "%.16e") + ",";
        res += number2string(size_MB, "%.16e") + ",";
        return res;
    }

    std::string str() {
        std::string res = "";
        res += mode + "  ";
        res += std::to_string(min_level) + "  ";
        res += std::to_string(max_level) + "  ";
        res += std::to_string(nx) + "  ";
        res += std::to_string(ny) + "  ";
        res += std::to_string(effective_resolution) + "  ";
        res += std::to_string(nDOFs) + "  ";
        res += std::to_string(lI_error) + "  ";
        res += std::to_string(l1_error) + "  ";
        res += std::to_string(l2_error) + "  ";
        res += std::to_string(build_time) + "  ";
        res += std::to_string(upwards_time) + "  ";
        res += std::to_string(solve_time) + "  ";
        res += std::to_string(size_MB) + "  ";
        return res;
    }

    static std::string headers() {
        std::string res = "";
        res += "mode,";
        res += "min_level,";
        res += "max_level,";
        res += "nx,";
        res += "ny,";
        res += "effective_resolution,";
        res += "nDOFs,";
        res += "lI_error,";
        res += "l1_error,";
        res += "l2_error,";
        res += "build_time,";
        res += "upwards_time,";
        res += "solve_time,";
        res += "size_MB,";
        return res;
    }

};

/**
 * @brief Class for the Gaussian Possion Problem
 * 
 * Solves the following BVP
 * 
 *      PDE: \nabla^2 u(x,y) = f(x,y), x \in \Omega = [-1, 1]^2
 *      BC:  u(x,y) = g_D(x,y), x \in \Gamma_D
 *      BC:  u(x,y) = g_N(x,y), x \in \Gamma_N
 * 
 * This problem has the exact solution:
 *      u(x,y) = exp(-(sigma_x*pow(x - x0, 2) + sigma_y*pow(y - y0, 2)))
 */
class GaussianPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    double x0 = 0;
    double y0 = 0;
    double sigma_x = 1;
    double sigma_y = 1;

    GaussianPoissonProblem() {}
    
    GaussianPoissonProblem(double x0, double y0, double sigma_x, double sigma_y) :
        x0(x0),
        y0(y0),
        sigma_x(sigma_x),
        sigma_y(sigma_y)
            {}

    std::string name() override { return "gaussian"; }

    double lambda() { return 0.0; }

    double u(double x, double y) override {
        return exp(-(sigma_x*pow(x - x0, 2) + sigma_y*pow(y - y0, 2)));
    }

    double f(double x, double y) override {
        double s = u(x,y);
        return s*(-2.0*sigma_x - 2.0*sigma_y + 4.0*pow((x - x0)*sigma_x, 2) + 4.0*pow((y - y0)*sigma_y, 2));
    }

    double dudx(double x, double y) override {
        return -2.0*u(x, y)*(x - x0)*sigma_x;
    }

    double dudy(double x, double y) override {
        return -2.0*u(x, y)*(y - y0)*sigma_y;
    }

};

/**
 * @brief Class for the Egg Carton Poisson Problem
 * 
 * Solves the following BVP
 * 
 *      PDE: \nabla^2 u(x,y) = f(x,y), x \in \Omega = [-1, 1]^2
 *      BC:  u(x,y) = g_D(x,y), x \in \Gamma_D
 *      BC:  u(x,y) = g_N(x,y), x \in \Gamma_N
 * 
 * This problem has the exact solution:
 *      u(x,y) = sin(4.0*M_PI*x) + cos(4.0*M_PI*y);
 */
class EggCartonPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    EggCartonPoissonProblem() {}

    std::string name() override { return "egg-carton"; }

    double lambda() { return 0.0; }

    double u(double x, double y) override {
        // return sin(2.0*M_PI*x) * sinh(2.0*M_PI*y);
        // return sin(4.0*M_PI*x) + cos(4.0*M_PI*y);
        // return jn(0, 40.0*sqrt(pow(x+2,2) + pow(y,2)));
        return sin(x) + sin(y);
    }

    double f(double x, double y) override {
        // return 0.0;
        // return -16.0*pow(M_PI, 2)*(cos(4.0*M_PI*y) + sin(4.0*M_PI*x));
        // return 0.0;
        return pow(cos(y),2)*sin(x) + pow(cos(x),2)*sin(y) - sin(x)*(2 + sin(x)*sin(y)) - sin(y)*(2 + sin(x)*sin(y));
    }

    double dudx(double x, double y) override {
        return 4.0*M_PI*cos(4.0*M_PI*x);
    }

    double dudy(double x, double y) override {
        return -4.0*M_PI*sin(4.0*M_PI*y);
    }

};

/**
 * @brief Class for the Polar Star Poisson Problem
 * 
 * Solves the following BVP
 * 
 *      PDE: \nabla^2 u(x,y) = f(x,y), x \in \Omega = [-1, 1]^2
 *      BC:  u(x,y) = g_D(x,y), x \in \Gamma_D
 *      BC:  u(x,y) = g_N(x,y), x \in \Gamma_N
 * 
 * The exact solution for this problem is long and complicated, but the solution is a set of polar
 * stars with high curavture (RHS) along the boundary of the polar star; ideal for an adaptive
 * mesh implementation.
 */
class PolarStarPoissonProblem : public EllipticForest::FISHPACK::FISHPACKProblem {

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

    std::string name() override { return "polar_star"; }

    double lambda() { return 0.0; }

    double u(double x, double y) override {
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

    double f(double x, double y) override {
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

    double dudx(double x, double y) override {
        return 0;
    }

    double dudy(double x, double y) override {
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
 * @brief Does a single solve of the Poisson problem via HPS method
 * 
 * @param pde The problem to solve
 * @param vtkFlag Flag to output VTK files
 * @return ResultsData Data structure with results of solver
 */
ResultsData solvePoissonViaHPS(EllipticForest::FISHPACK::FISHPACKProblem& pde, bool vtkFlag) {

    // Get the options
    EllipticForest::EllipticForestApp& app = EllipticForest::EllipticForestApp::getInstance();
    int minLevel = std::get<int>(app.options["min-level"]);
    int maxLevel = std::get<int>(app.options["max-level"]);
    int nx = std::get<int>(app.options["nx"]);
    int ny = std::get<int>(app.options["ny"]);
    std::string mode = minLevel == maxLevel ? "uniform" : "adaptive";

    // Create p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, 0, minLevel, fillUniform, 0, NULL, NULL);
    p4est->user_pointer = &pde;

    // Refine the p4est according to the RHS up to the max level
    p4est_refine(p4est, refineRecursive,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        // Get app context
        auto& pde = *((EllipticForest::FISHPACK::FISHPACKProblem*) p4est->user_pointer);
        auto& app = EllipticForest::EllipticForestApp::getInstance();
        int maxLevel = std::get<int>(app.options["max-level"]);
        int nx = std::get<int>(app.options["nx"]);
        int ny = std::get<int>(app.options["ny"]);
        double threshold = std::get<double>(app.options["refinement-threshold"]);

        // Do not refine if at the max level
        if (quadrant->level >= maxLevel) {
            return 0;
        }

        // Get bounds of quadrant
        double vxyz[3];
        double xLower, xUpper, yLower, yUpper;
        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, vxyz);
        xLower = vxyz[0];
        yLower = vxyz[1];

        p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), vxyz);
        xUpper = vxyz[0];
        yUpper = vxyz[1];

        // if (yLower < -0.9 || yUpper > 0.9) {
        //     return 1;
        // }
        // else {
        //     return 0;
        // }

        // Create quadrant grid
        EllipticForest::Petsc::PetscGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
        // EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);

        // Iterate over grid and check for refinement threshold
        for (auto i = 0; i < nx; i++) {
            double x = grid(XDIM, i);
            for (auto j = 0; j < ny; j++) {
                double y = grid(YDIM, j);
                double f = pde.f(x,y);
                if (fabs(f) > threshold) {
                    return 1;
                }
            }
        }

        return 0;
    },
    NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);
    p4est_partition(p4est, 0, NULL);

    // Save initial mesh
    if (vtkFlag) {
        std::string VTKFilename = "poisson_mesh_" + mode + "_" + pde.name();
        p4est_vtk_context_t* vtk_context = p4est_vtk_context_new(p4est, VTKFilename.c_str());
        p4est_vtk_context_set_scale(vtk_context, 1.0);
        vtk_context = p4est_vtk_write_header(vtk_context);
        p4est_vtk_write_footer(vtk_context);
        // p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());
    }

    // Create leaf level root patch
    double xLower = -1;
    double xUpper = 1;
    double yLower = -1;
    double yUpper = 1;
    EllipticForest::Petsc::PetscGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    // EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::Petsc::PetscPatch rootPatch(grid);
    // EllipticForest::FISHPACK::FISHPACKPatch rootPatch(grid);
    // rootPatch.level = 0;
    // rootPatch.isLeaf = true;

    // Create patch solver
    EllipticForest::Petsc::PetscPatchSolver solver{};
    solver.setAlphaFunction([&](double x, double y){
        return 1.0;
    });
    solver.setBetaFunction([&](double x, double y){
        return 1.0;
    });
    solver.setLambdaFunction([&](double x, double y){
        return 0.0;
    });
    // EllipticForest::FISHPACK::FISHPACKFVSolver solver{};

    // Create node factory
    EllipticForest::Petsc::PetscPatchNodeFactory nodeFactory;
    // EllipticForest::FISHPACK::FISHPACKPatchNodeFactory nodeFactory;

    // Create and run HPS method
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::Petsc::PetscGrid,
        EllipticForest::Petsc::PetscPatchSolver,
        EllipticForest::Petsc::PetscPatch,
        double>
            HPS(MPI_COMM_WORLD, p4est, rootPatch, solver, &nodeFactory);
    // EllipticForest::HPSAlgorithm
    //     <EllipticForest::FISHPACK::FISHPACKFVGrid,
    //     EllipticForest::FISHPACK::FISHPACKFVSolver,
    //     EllipticForest::FISHPACK::FISHPACKPatch,
    //     double>
    //         HPS(MPI_COMM_WORLD, p4est, rootPatch, solver, &nodeFactory);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    int nSolves = 1;
    for (auto n = 0; n < nSolves; n++) {
        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            HPS.upwardsStage([&](EllipticForest::Petsc::PetscPatch& leafPatch){
            // HPS.upwardsStage([&](EllipticForest::FISHPACK::FISHPACKPatch& leafPatch){
                EllipticForest::Petsc::PetscGrid& grid = leafPatch.grid();
                // EllipticForest::FISHPACK::FISHPACKFVGrid& grid = leafPatch.grid();
                leafPatch.vectorF() = EllipticForest::Vector<double>(grid.nPointsX() * grid.nPointsY());
                for (auto i = 0; i < grid.nPointsX(); i++) {
                    double x = grid(0, i);
                    for (auto j = 0; j < grid.nPointsY(); j++) {
                        double y = grid(1, j);
                        int index = j + i*grid.nPointsY();
                        leafPatch.vectorF()[index] = pde.f(x, y);
                    }
                }
                return;
            });
        }

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            switch (side) {
                case 0:
                    // West : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 1:
                    // East : Dirichlet
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 2:
                    // South : Neumann
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);

                case 3:
                    // North : Neumann
                    *a = 1.0;
                    *b = 0.0;
                    return pde.u(x,y);
                
                default:
                    break;
            }

            return 0.0;
        });
    }

    // Output mesh
    if (vtkFlag) {

        EllipticForest::Mesh<EllipticForest::Petsc::PetscPatch> mesh{HPS.quadtree};
        EllipticForest::Vector<double> uMesh{};
        EllipticForest::Vector<double> fMesh{};
        mesh.quadtree->traversePreOrder([&](EllipticForest::Node<EllipticForest::Petsc::PetscPatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                uMesh.append(patch.vectorU());
                fMesh.append(patch.vectorF());
            }
            return 1;
        });
        uMesh.name() = "u_soln";
        fMesh.name() = "f_rhs";

        EllipticForest::UnstructuredGridVTK vtu{};
        vtu.buildMesh(mesh);
        vtu.addCellData(uMesh);
        vtu.addCellData(fMesh);
        vtu.toVTK("dummy.vtu");

    }

    // Compute error of solution
    double l1_error = 0;
    double l2_error = 0;
    double lI_error = 0;
    int nLeafPatches = 0;
    HPS.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::Petsc::PetscPatch>* node){
    // HPS.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::FISHPACK::FISHPACKPatch>* node){
        if (node->leaf) {
            EllipticForest::Petsc::PetscPatch& patch = node->data;
            EllipticForest::Petsc::PetscGrid& grid = patch.grid();
            // EllipticForest::FISHPACK::FISHPACKPatch& patch = node->data;
            // EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid();
            for (auto i = 0; i < grid.nPointsX(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.nPointsY(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.nPointsY();
                    int index_T = i + j*grid.nPointsY();
                    // double un = patch.vectorU()[index];
                    // double ue = pde.u(x,y);
                    // double diff = un - ue;
                    // printf("i = %4i, j = %4i, x = %8.4f, y = %8.4f\n", i, j, x, y);
                    // patch.vectorU()[index] = ue;
                    // app.log("un = %12.4f    ue = %12.4f    diff = %12.4e", un, ue, diff);
                    double diff = patch.vectorU()[index] - pde.u(x, y);
                    l1_error += (grid.dx()*grid.dy())*fabs(diff);
                    l2_error += (grid.dx()*grid.dy())*pow(fabs(diff), 2);
                    lI_error = fmax(lI_error, fabs(diff));
                }
            }
            nLeafPatches++;
        }
        return 1;
    });
    double area = (xUpper - xLower) * (yUpper - yLower);
    l1_error = l1_error / area;
    l2_error = sqrt(l2_error / area);
    int resolution = pow(2,maxLevel)*nx;
    int nDOFs = nLeafPatches * (nx * ny);

    // Compute size of quadtree and data
    double size_MB = 0;
    // HPS.quadtree.traversePostOrder([&](EllipticForest::Petsc::PetscPatch& patch){
    //     size_MB += patch.dataSize();
    // });

    // Store and return results
    ResultsData results;
    results.mode = mode;
    results.min_level = minLevel;
    results.max_level = maxLevel;
    results.nx = nx;
    results.ny = ny;
    results.effective_resolution = resolution;
    results.nDOFs = nDOFs;
    results.l1_error = l1_error;
    results.l2_error = l2_error;
    results.lI_error = lI_error;
    results.build_time = app.timers["build-stage"].time();
    results.upwards_time = app.timers["upwards-stage"].time();
    results.solve_time = app.timers["solve-stage"].time();
    results.size_MB = size_MB;

    // PyObject* pyObj;
    // HPS.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::Petsc::PetscPatch>* node){
    // // HPS.quadtree.traversePostOrder([&](EllipticForest::Node<EllipticForest::FISHPACK::FISHPACKPatch>* node){
    //     if (node->leaf) {
    //         EllipticForest::Petsc::PetscPatch& patch = node->data;
    //         EllipticForest::Petsc::PetscGrid& grid = patch.grid();
    //         // EllipticForest::FISHPACK::FISHPACKPatch& patch = node->data;
    //         // EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid();
    //         EllipticForest::Vector<float> uFloat(patch.vectorU().size());
    //         for (int i = 0; i < uFloat.size(); i++) {
    //             uFloat[i] = static_cast<float>(patch.vectorU()[i]);
    //         }
    //         plt::imshow(uFloat.dataPointer(), grid.nPointsX(), grid.nPointsY(), 1, {}, &pyObj);
    //         plt::colorbar(pyObj);
    //         plt::title(std::to_string(grid.nPointsX()) + ": [" + std::to_string(grid.xLower()) + "," + std::to_string(grid.xUpper()) + "] x [" + std::to_string(grid.yLower()) + "," + std::to_string(grid.yUpper()) + "]");
    //         plt::show();
    //     }
    //     return 1;
    // });

    return results;

}

int main(int argc, char** argv) {

    // Initialize app
    EllipticForest::EllipticForestApp app(&argc, &argv);
    EllipticForest::MPI::MPIObject mpi(MPI_COMM_WORLD);

    // Set options
    app.options.setOption("cache-operators", false);
    app.options.setOption("homogeneous-rhs", false);
    app.options.setOption("refinement-threshold", 2.0);

    // Create PDE to solve
    // PolarStarPoissonProblem pde(
    //     2,              // Number of polar stars
    //     {-0.5, 0.5, 0.5, -0.5},     // x0
    //     {-0.5, -0.5, 0.5, 0.5},     // y0
    //     {0.1, 0.2, 0.3, 0.4},       // r0
    //     {0.2, 0.3, 0.4, 0.5},       // r1
    //     {3, 4, 5, 8},               // n
    //     0.001                       // epsilon
    // );
    // GaussianPoissonProblem pde(
    //     0,            // x0
    //     0,            // y0
    //     80,              // sigma_x
    //     10               // sigma_y
    // );
    EggCartonPoissonProblem pde{};

    // Convergence sweep
    std::vector<int> patchSizeVector = {32};     // Size of patch
    std::vector<int> levelVector {5};              // Maximum level of refinement (uniform: L-L, adaptive: 0-L)

    // Create storage for plotting
    std::vector<PlotPair> uniformErrorPlots;
    std::vector<PlotPair> uniformBuildTimingPlots;
    std::vector<PlotPair> uniformSolveTimingPlots;
    std::vector<PlotPair> adaptiveErrorPlots;
    std::vector<PlotPair> adaptiveBuildTimingPlots;
    std::vector<PlotPair> adaptiveSolveTimingPlots;

    // Vector of results
    std::vector<ResultsData> resultsVector;

    // Run uniform parameter sweep
    bool vtkFlag = true;
    int maxResolution = pow(128, 2) * pow(2, 2*5); // About 16M DOFs
    for (auto& M : patchSizeVector) {

        PlotPair errorPair;
        PlotPair buildPair;
        PlotPair solvePair;

        for (auto& l : levelVector) {

            app.logHead("UNIFORM: M = %i, l = %i", M, l);
            int DOFs = pow(M, 2) * pow(2, 2*l);
            if (DOFs >= maxResolution) {
                app.log("Skipping...");
                continue;
            }

            // Set options
            app.options.setOption("min-level", 0);
            app.options.setOption("max-level", l);
            app.options.setOption("nx", M);
            app.options.setOption("ny", M);

            // Solve via HPS
            if (M == 128 && l == 4) vtkFlag = true;
            else vtkFlag = false;
            vtkFlag = true;
            ResultsData results = solvePoissonViaHPS(pde, vtkFlag);
            int nDOFs = results.effective_resolution;
            double error = results.lI_error;
            resultsVector.push_back(results);

            // Save info to plot
            errorPair.first.push_back(nDOFs);
            errorPair.second.push_back(error);

            buildPair.first.push_back(nDOFs);
            buildPair.second.push_back(app.timers["build-stage"].time());

            solvePair.first.push_back(nDOFs);
            solvePair.second.push_back(app.timers["solve-stage"].time());

            // Restart timers
            app.timers["build-stage"].restart();
            app.timers["upwards-stage"].restart();
            app.timers["solve-stage"].restart();
        }

        uniformErrorPlots.push_back(errorPair);
        uniformBuildTimingPlots.push_back(buildPair);
        uniformSolveTimingPlots.push_back(solvePair);
    }

    // Run adaptive parameter sweep
    // for (auto& M : patchSizeVector) {

    //     PlotPair errorPair;
    //     PlotPair buildPair;
    //     PlotPair solvePair;

    //     for (auto& l : levelVector) {

    //         app.log("ADAPTIVE: M = %i, l = %i", M, l);
    //         int DOFs = pow(M, 2) * pow(2, 2*l);
    //         if (DOFs >= maxResolution) {
    //             app.log("Skipping...");
    //             continue;
    //         }

    //         // Set options
    //         app.options.setOption("min-level", 0);
    //         app.options.setOption("max-level", l);
    //         app.options.setOption("nx", M);
    //         app.options.setOption("ny", M);

    //         // Solve via HPS
    //         if (M == 128 && l == 4) vtkFlag = true;
    //         else vtkFlag = false;
    //         ResultsData results = solvePoissonViaHPS(pde, vtkFlag);
    //         int nDOFs = results.effective_resolution;
    //         double error = results.lI_error;
    //         resultsVector.push_back(results);

    //         // Save info to plot
    //         errorPair.first.push_back(nDOFs);
    //         errorPair.second.push_back(error);

    //         buildPair.first.push_back(nDOFs);
    //         buildPair.second.push_back(app.timers["build-stage"].time());

    //         solvePair.first.push_back(nDOFs);
    //         solvePair.second.push_back(app.timers["solve-stage"].time());

    //         // Restart timers
    //         app.timers["build-stage"].restart();
    //         app.timers["upwards-stage"].restart();
    //         app.timers["solve-stage"].restart();
    //     }

    //     adaptiveErrorPlots.push_back(errorPair);
    //     adaptiveBuildTimingPlots.push_back(buildPair);
    //     adaptiveSolveTimingPlots.push_back(solvePair);
    // }

    // Write results to console
    app.log(ResultsData::headers());
    for (auto& results : resultsVector) {
        app.log(results.str());
    }

    // Write results to file
    std::ofstream csvFile;
    csvFile.open("poisson_results.csv");
    csvFile << ResultsData::headers() << std::endl;
    for (auto& results : resultsVector) {
        csvFile << results.csv() << std::endl;
    }
    csvFile.close();
    
    #ifdef USE_MATPLOTLIBCPP
    if (mpi.getRank() == EllipticForest::MPI::HEAD_RANK) {
        // Error plot
        int fig1 = plt::figure(1);
        int counter = 0;
        std::vector<std::string> colors = {"r", "g", "b", "y", "c", "m"};
        for (auto& [nDOFs, error] : uniformErrorPlots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "--s" + colors[counter]);
            counter++;
        }
        counter = 0;
        // for (auto& [nDOFs, error] : adaptiveErrorPlots) {
        //     plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, error, "-o" + colors[counter]);
        //     counter++;
        // }
        std::vector<int> xTicks = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        std::vector<std::string> xTickLabels;
        for (auto& t : xTicks) xTickLabels.push_back(std::to_string(t));
        plt::xlabel("Effective Resolution");
        plt::ylabel("Inf-Norm Error");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "upper right"}});
        plt::grid(true);
        plt::save("plot_poisson_error_" + pde.name() + "_no_title.pdf");
        plt::title("Convergence Study - Uniform vs. Adaptive Mesh");
        plt::save("plot_poisson_error_" + pde.name() + ".pdf");
        plt::show();

        int fig2 = plt::figure(2);
        counter = 0;
        for (auto& [nDOFs, build] : uniformBuildTimingPlots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "--s" + colors[counter]);
            counter++;
        }
        counter = 0;
        for (auto& [nDOFs, build] : adaptiveBuildTimingPlots) {
            plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, build, "-o" + colors[counter]);
            counter++;
        }
        plt::xlabel("Effective Resolution");
        plt::ylabel("Time [sec]");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "lower right"}});
        plt::grid(true);
        plt::save("plot_poisson_build_time_" + pde.name() + "_no_title.pdf");
        plt::title("Timing Study - Uniform vs. Adaptive Mesh - Build Stage");
        plt::save("plot_poisson_build_time_" + pde.name() + ".pdf");
        plt::show();

        int fig3 = plt::figure(3);
        counter = 0;
        for (auto& [nDOFs, solve] : uniformSolveTimingPlots) {
            plt::named_loglog("Uniform: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "--s" + colors[counter]);
            counter++;
        }
        counter = 0;
        for (auto& [nDOFs, solve] : adaptiveSolveTimingPlots) {
            plt::named_loglog("Adaptive: N = " + std::to_string(patchSizeVector[counter]), nDOFs, solve, "-o" + colors[counter]);
            counter++;
        }
        plt::xlabel("Effective Resolution");
        plt::ylabel("Time [sec]");
        plt::xticks(xTicks, xTickLabels);
        plt::legend({{"loc", "lower right"}});
        plt::grid(true);
        plt::save("plot_poisson_solve_time_" + pde.name() + "_no_title.pdf");
        plt::title("Timing Study - Uniform vs. Adaptive Mesh - Solve Stage");
        plt::save("plot_poisson_solve_time_" + pde.name() + ".pdf");
        plt::show();
    }
    #endif

    return EXIT_SUCCESS;
}