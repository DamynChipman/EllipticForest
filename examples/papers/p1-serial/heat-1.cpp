#include <iostream>

#include <EllipticForest.hpp>

#include "common.hpp"

class PoissonProblem1 : public EllipticForest::FISHPACK::FISHPACKProblem {

public:

    double c;

    PoissonProblem1(double c) :
        c(c)
            {}

    std::string name() override { return "poisson-1"; }

    double lambda() { return 0.0; }

    double u(double x, double y) {
        return sin(c*M_PI*x) + cos(c*M_PI*y);
    }

    double f(double x, double y) {
        return -pow(c*M_PI, 2)*(sin(c*M_PI*x) + cos(c*M_PI*y));
    }

    double dudx(double x, double y) {
        return c*M_PI*cos(c*M_PI*x);
    }

    double dudy(double x, double y) {
        return -c*M_PI*sin(c*M_PI*y);
    }

};

double computeExactSolution(EllipticForest::HPSAlgorithm<EllipticForest::FISHPACK::FISHPACKFVGrid, EllipticForest::FISHPACK::FISHPACKFVSolver, EllipticForest::FISHPACK::FISHPACKPatch, double>& HPS, EllipticForest::FISHPACK::FISHPACKProblem& pde, double x, double y, double time) {

    auto& quadtree = HPS.mesh.quadtree;

    double uExact = 0;
    int M = 100;
    int N = 100;
    for (int m = 1; m <= M; m++) {
        for (int n = 1; n <= N; n++) {

            double A_mn = (4*(4*m*M_PI*(pow(n,2) - 4*pow(M_PI,2))*cos(2*pow(M_PI,2))*sin(m*M_PI)*pow(sin((n*M_PI)/2.),2) + n*(pow(m,2) - 4*pow(M_PI,2))*(n - n*cos(n*M_PI)*cos(2*pow(M_PI,2)) - 2*M_PI*sin(n*M_PI)*sin(2*pow(M_PI,2))) + cos(m*M_PI)*(-(pow(m,2)*(-pow(n,2) + 4*pow(M_PI,2))*(-1 + cos(n*M_PI))*sin(2*pow(M_PI,2))) - n*(-pow(m,2) + 4*pow(M_PI,2))*(-n + n*cos(n*M_PI)*cos(2*pow(M_PI,2)) + 2*M_PI*sin(n*M_PI)*sin(2*pow(M_PI,2))))))/(m*n*pow(M_PI,2)*(pow(m,2) - 4*pow(M_PI,2))*(pow(n,2) - 4*pow(M_PI,2)));

            // Compute sum
            double kx = m*M_PI;
            double ky = n*M_PI;
            uExact += A_mn * sin(kx*x) * sin(ky*y) * exp(-(pow(kx,2) + pow(ky,2))*time);
        }
    }

    return uExact;

}

void run(EllipticForest::FISHPACK::FISHPACKProblem& pde) {

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

        // Create quadrant grid
        EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);

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

    // Create leaf level root patch
    double xLower = 0;
    double xUpper = 1;
    double yLower = 0;
    double yUpper = 1;
    EllipticForest::FISHPACK::FISHPACKFVGrid grid(nx, ny, xLower, xUpper, yLower, yUpper);
    EllipticForest::FISHPACK::FISHPACKPatch rootPatch(grid);
    rootPatch.level = 0;
    rootPatch.isLeaf = true;

    // Create time data
    double tI = 0;
    double tF = 0.01;
    int nt = 10;
    double dt = (tF - tI) / (nt - 1);
    double lambda = -1.0/dt;

    // Create patch solver
    EllipticForest::FISHPACK::FISHPACKFVSolver solver(lambda);

    // Create node factory
    EllipticForest::FISHPACK::FISHPACKPatchNodeFactory nodeFactory{};

    // Create mesh from p4est and patch solver
    EllipticForest::Mesh<EllipticForest::FISHPACK::FISHPACKPatch> mesh{MPI_COMM_WORLD, p4est, rootPatch, nodeFactory};

    // Create and run HPS method
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm
        <EllipticForest::FISHPACK::FISHPACKFVGrid,
        EllipticForest::FISHPACK::FISHPACKFVSolver,
        EllipticForest::FISHPACK::FISHPACKPatch,
        double>
            HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop
    std::vector<ResultsData> resultsVector;
    for (auto n = 0; n < nt; n++) {

        // Compute time
        double time = tI + n*dt;

        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        if (n == 0) {
            // Setup initial condition on RHS
            HPS.upwardsStage([&](EllipticForest::FISHPACK::FISHPACKPatch& leafPatch){
                EllipticForest::FISHPACK::FISHPACKFVGrid& grid = leafPatch.grid();
                leafPatch.vectorF() = EllipticForest::Vector<double>(grid.nPointsX() * grid.nPointsY());
                for (auto i = 0; i < grid.nPointsX(); i++) {
                    double x = grid(0, i);
                    for (auto j = 0; j < grid.nPointsY(); j++) {
                        double y = grid(1, j);
                        int index = j + i*grid.nPointsY();
                        leafPatch.vectorF()[index] = pde.u(x, y);
                    }
                }
                return;
            });
        }
        else {
            // Update RHS with previous solution
            HPS.upwardsStage([&](EllipticForest::FISHPACK::FISHPACKPatch& leafPatch){
                leafPatch.vectorF() = lambda*leafPatch.vectorU();
            });
        }

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return 0.0;
        });

        // Compute error
        app.log("Computing error...");
        double l1_error = 0;
        double l2_error = 0;
        double lI_error = 0;
        int nLeafPatches = 0;
        mesh.quadtree.traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
            if (patch.isLeaf) {
                EllipticForest::FISHPACK::FISHPACKFVGrid& grid = patch.grid();
                for (auto i = 0; i < grid.nPointsX(); i++) {
                    double x = grid(XDIM, i);
                    for (auto j = 0; j < grid.nPointsY(); j++) {
                        double y = grid(YDIM, j);
                        double uExact = computeExactSolution(HPS, pde, x, y, time);
                        int index = j + i*grid.nPointsY();
                        int index_T = i + j*grid.nPointsY();
                        double diff = patch.vectorU()[index_T] - uExact;
                        l1_error += (grid.dx()*grid.dy())*fabs(diff);
                        l2_error += (grid.dx()*grid.dy())*pow(fabs(diff), 2);
                        lI_error = fmax(lI_error, fabs(diff));
                    }
                }
                nLeafPatches++;
            }
        });
        double area = (xUpper - xLower) * (yUpper - yLower);
        l1_error = l1_error / area;
        l2_error = sqrt(l2_error / area);
        int resolution = pow(2,maxLevel)*nx;
        int nDOFs = nLeafPatches * (nx * ny);

        app.log("Time = %f, LI_error = %11.4e, L1_error = %11.4e, L2_error = %11.4e", time, lI_error, l1_error, l2_error);

        // Compute size of quadtree and data
        double size_MB = 0;
        mesh.quadtree.traversePostOrder([&](EllipticForest::FISHPACK::FISHPACKPatch& patch){
            size_MB += patch.dataSize();
        });

        // Store and return results
        ResultsData results;
        results.mode = mode;
        results.min_level = minLevel;
        results.max_level = maxLevel;
        results.nx = nx;
        results.ny = ny;
        results.effective_resolution = resolution;
        results.nDOFs = nDOFs;
        results.lambda = lambda;
        results.time = time;
        results.l1_error = l1_error;
        results.l2_error = l2_error;
        results.lI_error = lI_error;
        results.build_time = app.timers["build-stage"].time();
        results.upwards_time = app.timers["upwards-stage"].time();
        app.timers["upwards-stage"].restart();
        results.solve_time = app.timers["solve-stage"].time();
        app.timers["solve-stage"].restart();
        results.size_MB = size_MB;

        resultsVector.push_back(results);

    }

    // Write results to console
    app.log(ResultsData::headers());
    for (auto& results : resultsVector) {
        app.log(results.str());
    }

    // Write results to file
    std::ofstream csvFile;
    csvFile.open("heat-1.csv");
    csvFile << ResultsData::headers() << std::endl;
    for (auto& results : resultsVector) {
        csvFile << results.csv() << std::endl;
    }
    csvFile.close();

}

int main(int argc, char** argv) {

    // Initialize app
    EllipticForest::EllipticForestApp app(&argc, &argv);

    // Set options
    app.options.setOption("cache-operators", true);
    app.options.setOption("homogeneous-rhs", false);
    app.options.setOption("refinement-threshold", 200.0);

    int patchSize = 32;
    int maxLevel = 4;
    int minLevel = 1;

    app.options.setOption("min-level", minLevel);
    app.options.setOption("max-level", maxLevel);
    app.options.setOption("nx", patchSize);
    app.options.setOption("ny", patchSize);

    // Create PDE for initial condition
    double c = 2*M_PI;
    PoissonProblem1 pde(c);

    // Iterate over time
    run(pde);

    return EXIT_SUCCESS;

}