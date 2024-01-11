/**
 * @file HPSAlgorithm.hpp
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Implementation of the quadtree-adaptive Hierarchical Poincare-Steklov (HPS) method as outlined in (Chipman, Calhoun, Burstedde, 2023)
 */
#ifndef HPS_ALGORITHM_HPP_
#define HPS_ALGORITHM_HPP_

#include <p4est.h>
#include "Quadtree.hpp"
#include "EllipticForestApp.hpp"
#include "DataCache.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"
#include "SpecialMatrices.hpp"
#include "Mesh.hpp"

namespace EllipticForest {

enum BoundaryConditionType {
    Dirichlet = 0,
    Neumann = 1,
    Robin = 2
};

template<typename PatchGridType, typename PatchSolverType, typename PatchType, typename NumericalType>
class HPSAlgorithm : public MPI::MPIObject {

public:

    /**
     * @brief Reference to the mesh
     * 
     */
    Mesh<PatchType>& mesh;

    /**
     * @brief The elliptic solver for an individual patch
     * 
     */
    PatchSolverType& patch_solver;

    /**
     * @brief Cache storage for matrices
     * 
     */
    DataCache<Matrix<NumericalType>> matrixCache{};

    /**
     * @brief Flag for if the build stage has already happened
     * 
     * TODO: Implement this logic
     * 
     */
    bool isBuilt = false;

    /**
     * @brief Construct a new HPSAlgorithm object
     * 
     * @param comm MPI communicator
     * @param rootPatch A patch representing the entire domain and leaf patch prototypes
     * @param patch_solver The elliptic solver for an individual patch
     * @param nodeFactory Node factory to produce new nodes
     */
    HPSAlgorithm(MPI::Communicator comm, p4est_t* p4est, PatchType& rootPatch, PatchSolverType& patch_solver, AbstractNodeFactory<PatchType>* nodeFactory) :
        MPIObject(comm),
        patch_solver(patch_solver),
        mesh(comm, p4est, rootPatch, nodeFactory)
            {}

    /**
     * @brief Construct a new HPSAlgorithm object
     * 
     * @param comm MPI communicator
     * @param mesh Pre-built refined mesh
     * @param patch_solver Patch solver
     */
    HPSAlgorithm(MPI::Communicator comm, Mesh<PatchType>& mesh, PatchSolverType& patch_solver) :
        MPIObject(comm),
        patch_solver(patch_solver),
        mesh(mesh)
            {}

    /**
     * @brief Performs the setup stage of the HPS method
     * 
     * NOTE: Do I need the setup stage anymore?
     * 
     * @param p4est The p4est object with the tree topology
     */
    virtual void setupStage() {
        
        // Get app and log
        MPI_Barrier(this->getComm());
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Setup Stage");

        app.addTimer("setup-stage");
        app.timers["setup-stage"].start();

        // NOTE: Nothing is currently done in setup stage... This might change.

        app.timers["setup-stage"].stop();
        app.logHead("End HPS Setup Stage");

    }

    /**
     * @brief Performs the build stage of the HPS method
     * 
     * The build stage just requires the tree topology, the patch solver's `buildD2N` method, and a patch grid. Iterates over the leaf nodes and constructs the
     * Dirichlet-to-Neumann matrix on each of the leaf nodes by calling the derived PatchSolver class function `buildD2N` and provides the leaf patch's grid.
     * After the leaf patches have the D2N matrices, recursively merges the patches up the tree (post-order) computing the set of solution matrices required
     * for the solve stage.
     * 
     * @sa PatchSolverBase.buildD2N
     * @sa merge4to1
     * 
     */
    virtual void buildStage() {

        MPI_Barrier(this->getComm());
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Build Stage");
        app.addTimer("build-stage");
        app.timers["build-stage"].start();

        mesh.quadtree.merge(
            [&](Node<PatchType>* leaf_node){
                // app.log("Leaf callback: path = " + leaf_node->path);
                // Leaf callback
                PatchType& patch = leaf_node->data;
                // patch.isLeaf = true;
                if (std::get<bool>(app.options["cache-operators"])) {
                    if (!matrixCache.contains("T_leaf")) {
                        matrixCache["T_leaf"] = patch_solver.buildD2N(patch.grid());
                    }
                    patch.matrixT() = matrixCache["T_leaf"];
                }
                else {
                    patch.matrixT() = patch_solver.buildD2N(patch.grid());
                }
                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // app.log("Family callback: parent path = " + parent_node->path);
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                merge4to1(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["build-stage"].stop();
        app.logHead("End HPS Build Stage");

    }

    /**
     * @brief Performs the upwards stage of the HPS method
     * 
     * The user provides a callback function `rhs_patch_function` that sets the leaf patch's load data (RHS data or non-homogeneous data). The callback function
     * provides a reference to a leaf patch. The callback function needs to set the `vectorF` data on the provided patch. This is done in post-order fashion for
     * all leaf patches. Next, the leaf patch's particular Neumann data is set (`vectorH`) through the base PatchSolver class function `particularNeumannData`,
     * which may or may not be overridden by the derived PatchSolver class.
     * 
     * @sa PatchBase.vectorF
     * @sa PatchBase.vectorH
     * @sa PatchSolverBase.particularNeumannData
     * @sa upwards4to1
     * 
     * @param rhs_patch_function Sets the leaf patch's load data `vectorF`
     */
    virtual void upwardsStage(std::function<void(PatchType& leafPatch)> rhs_patch_function) {

        MPI_Barrier(this->getComm());
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Upwards Stage");
        app.addTimer("upwards-stage");
        app.timers["upwards-stage"].start();

        mesh.quadtree.merge(
            [&](Node<PatchType>* leaf_node){
                // app.log("Leaf callback: path = " + leaf_node->path);
                // Leaf callback
                PatchType& patch = leaf_node->data;

                // Call callback function to set RHS data on patch
                rhs_patch_function(patch);

                // Set particular Neumann data using patch solver function
                patch.vectorH() = patch_solver.particularNeumannData(patch.grid(), patch.vectorF());

                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // app.log("Family callback: parent path = " + parent_node->path);
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                upwards4to1(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["upwards-stage"].stop();
        app.logHead("End HPS Upwards Stage");

    }

    /**
     * @brief Performs the upwards stage of the HPS method
     * 
     * The user provides a callback function `rhs_function` that is the analytical expression for the right-hand side of the PDE to solve.
     * 
     * @param rhs_function Analytical right-hand function
     */
    virtual void upwardsStage(std::function<double(double, double)> rhs_function) {

        MPI_Barrier(this->getComm());
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Upwards Stage");
        app.addTimer("upwards-stage");
        app.timers["upwards-stage"].start();

        mesh.quadtree.merge(
            [&](Node<PatchType>* leaf_node){
                // app.log("Leaf callback: path = " + leaf_node->path);
                // Leaf callback
                PatchType& patch = leaf_node->data;
                PatchGridType& grid = patch.grid();

                // Call callback function to set RHS data on patch
                patch.vectorF() = EllipticForest::Vector<double>(grid.nx() * grid.ny());
                for (auto i = 0; i < grid.nx(); i++) {
                    double x = grid(0, i);
                    for (auto j = 0; j < grid.ny(); j++) {
                        double y = grid(1, j);
                        int index = j + i*grid.ny();
                        patch.vectorF()[index] = rhs_function(x, y);
                    }
                }

                // Set particular Neumann data using patch solver function
                patch.vectorH() = patch_solver.particularNeumannData(grid, patch.vectorF());

                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // app.log("Family callback: parent path = " + parent_node->path);
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                upwards4to1(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["upwards-stage"].stop();
        app.logHead("End HPS Upwards Stage");

    }

    /**
     * @brief Performs the solve stage of the HPS method
     * 
     * The user provides a callback function `boundary_data_function` that sets the boundary condition data (currently just Dirichlet data) on the root patch, or
     * the physical boundary of the domain spanned by the tree. The callback function provides a reference to the root patch after all merging has been done
     * (so the grid on the provided root patch may not be the same as a leaf patch). The callback function needs to set the Dirichlet data `vectorG` on the
     * provided patch. Following the physical boundary data, the split function is called recursively down the tree (pre-order) applying the solution matrices
     * to the boundary data. This is done until all leaf patches have boundary data. Finally, patch solver is called on all leaf patches and the solution data
     * is set on the leaf patches (`vectorU`).
     * 
     * @sa PatchBase.vectorG
     * @sa PatchBase.vectorU
     * @sa PatchSolverBase.solve
     * @sa split1to4
     * 
     * @param boundary_data_function Sets the Dirichlet data `vectorG` on the provided patch
     */    
    virtual void solveStage(std::function<void(PatchType& rootPatch)> boundary_data_function) {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Solve Stage");
        app.addTimer("solve-stage");
        app.timers["solve-stage"].start();

        // Set Dirichlet data on root patch
        boundary_data_function(mesh.quadtree.root());

        mesh.quadtree.split(
            [&](Node<PatchType>* leaf_node){
                // app.log("Leaf callback: path = " + leaf_node->path);
                // Leaf callback
                leafSolve(leaf_node->data);
                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // app.log("Family callback: parent path = " + parent_node->path);
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                split1to4(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["solve-stage"].stop();
        app.logHead("End HPS Solve Stage");

    }

    /**
     * @brief Performs the solve stage of the HPS method
     * 
     * The user provides a callback function that has the analytical expression for the data at the boundary. The form of this function is
     * 
     *     `boundary_analytical_function(int side, NumericalType x, NumericalType y, NumericalType* a, NumericalType* b) -> NumericalType`.
     * 
     * The function must return the value of the boundary data for the given `side` and coordinates `x` and `y`. The function must also set the values of `a` and `b`
     * that correspond to the type of boundary data in the general boundary condition equation
     * 
     *     a(x,y) u(x,y) + b(x,y) dudn(x,y) = r(x,y)
     * 
     * for (x,y) at the boundary of the patch. The Dirichlet data needed for the solve stage is computed from the inverse of the Dirichlet-to-Neumann operator at the
     * root of the tree.
     * 
     * @param boundary_analytical_function Analytical expression for the boundary data as described above
     */
    virtual void solveStage(std::function<NumericalType(int side, NumericalType x, NumericalType y, NumericalType* a, NumericalType* b)> boundary_analytical_function) {

        MPI_Barrier(this->getComm());
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Solve Stage");
        app.addTimer("solve-stage");
        app.timers["solve-stage"].start();

        // Set up data for Dirichlet solve
        PatchType& rootPatch = mesh.quadtree.root();
        PatchGridType& rootGrid = rootPatch.grid();
        int M = rootGrid.nx();
        std::vector<Vector<NumericalType>> aVectors = {
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M)
        };
        std::vector<Vector<NumericalType>> bVectors = {
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M)
        };
        std::vector<Vector<NumericalType>> rVectors = {
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M),
            Vector<NumericalType>(M)
        };

        // Populate boundary data vectors
        for (auto n = 0; n < 4; n++) {
            for (auto i = 0; i < M; i++) {
                NumericalType x, y;
                if (n == 0) {
                    x = rootGrid.xLower();
                    y = rootGrid(1, i);
                }
                else if (n == 1) {
                    x = rootGrid.xUpper();
                    y = rootGrid(1, i);
                }
                else if (n == 2) {
                    x = rootGrid(0, i);
                    y = rootGrid.yLower();
                }
                else if (n == 3) {
                    x = rootGrid(0, i);
                    y = rootGrid.yUpper();
                }

                NumericalType a, b;
                rVectors[n][i] = boundary_analytical_function(n, x, y, &a, &b);
                aVectors[n][i] = a;
                bVectors[n][i] = b;
            }
        }

        // Solve for Dirichlet data : g = (a*I + b*I*T)^(-1) * (r - b*I*h)
        {
            Vector<NumericalType> r = concatenate(rVectors);
            Vector<NumericalType> a = concatenate(aVectors);
            Vector<NumericalType> b = concatenate(bVectors);
            Vector<NumericalType>& h = rootPatch.vectorH();
            Matrix<NumericalType>& T = rootPatch.matrixT();
            Vector<NumericalType>& g = rootPatch.vectorG();

            Matrix<NumericalType> A = DiagonalMatrix<NumericalType>(a);
            Matrix<NumericalType> B = DiagonalMatrix<NumericalType>(b);
            Vector<NumericalType> bh = B * h;
            Vector<NumericalType> rhs = r - bh;

            B = B * T;
            A = A + B;

            g = solve(A, rhs);
        }

        mesh.quadtree.split(
            [&](Node<PatchType>* leaf_node){
                // app.log("Leaf callback: path = " + leaf_node->path);
                // Leaf callback
                leafSolve(leaf_node->data);
                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // app.log("Family callback: parent path = " + parent_node->path);
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                split1to4(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["solve-stage"].stop();
        app.logHead("End HPS Solve Stage");
                
    }

// I believe this is depreciated
#if 0 
    virtual void solveStage(std::vector<Vector<NumericalType>> boundaryData, std::vector<BoundaryConditionType> boundaryTypes) {
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.logHead("Begin HPS Solve Stage");
        app.addTimer("solve-stage");
        app.timers["solve-stage"].start();

        setBoundaryData_(mesh.quadtree.root(), boundaryData, boundaryTypes);

        // mesh.quadtree.split([&](PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega){
        //     split1to4(tau, alpha, beta, gamma, omega);
        // });

        // mesh.quadtree.traversePreOrder([&](PatchType& patch){
        //     leafSolve(patch);
        // });
        mesh.quadtree.split(
            [&](Node<PatchType>* leaf_node){
                // Leaf callback
                leafSolve(leaf_node->data);
                return 1;
            },
            [&](Node<PatchType>* parent_node, std::vector<Node<PatchType>*> child_nodes){
                // Family callback
                PatchType& tau = parent_node->data;
                PatchType& alpha = child_nodes[0]->data;
                PatchType& beta = child_nodes[1]->data;
                PatchType& gamma = child_nodes[2]->data;
                PatchType& omega = child_nodes[3]->data;
                split1to4(tau, alpha, beta, gamma, omega);
                return 1;
            }
        );

        app.timers["solve-stage"].stop();
        app.logHead("End HPS Solve Stage");

    }
#endif

    /**
     * @brief Recursive merge function
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    virtual void merge4to1(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        // app.logHead("Merging:");
        // app.logHead("  alpha = %i", alpha.globalID);
        // app.logHead("  beta = %i", beta.globalID);
        // app.logHead("  gamma = %i", gamma.globalID);
        // app.logHead("  omega = %i", omega.globalID);
        // app.logHead("  tau = %i", tau.globalID);

        // Steps for the merge (private member functions)
        coarsen_(tau, alpha, beta, gamma, omega);
        createIndexSets_(tau, alpha, beta, gamma, omega);
        createMatrixBlocks_(tau, alpha, beta, gamma, omega);
        mergeX_(tau, alpha, beta, gamma, omega);
        mergeS_(tau, alpha, beta, gamma, omega);
        mergeT_(tau, alpha, beta, gamma, omega);
        reorderOperators_(tau, alpha, beta, gamma, omega);
        mergePatch_(tau, alpha, beta, gamma, omega);

        return;
    }

    static void merge4to1ExternalInterface(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        // app.logHead("Merging:");
        // app.logHead("  alpha = %i", alpha.globalID);
        // app.logHead("  beta = %i", beta.globalID);
        // app.logHead("  gamma = %i", gamma.globalID);
        // app.logHead("  omega = %i", omega.globalID);
        // app.logHead("  tau = %i", tau.globalID);

        // Steps for the merge (private member functions)
        coarsen(tau, alpha, beta, gamma, omega);
        mergeX(tau, alpha, beta, gamma, omega);
        mergeS(tau, alpha, beta, gamma, omega);
        mergeT(tau, alpha, beta, gamma, omega);
        reorderOperators(tau, alpha, beta, gamma, omega);
        mergePatch(tau, alpha, beta, gamma, omega);

        return;
    }

    /**
     * @brief Recursive upwards function
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    virtual void upwards4to1(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            // app.logHead("Upwards:");
            // app.logHead("  alpha = %i", alpha.globalID);
            // app.logHead("  beta = %i", beta.globalID);
            // app.logHead("  gamma = %i", gamma.globalID);
            // app.logHead("  omega = %i", omega.globalID);
            // app.logHead("  tau = %i", tau.globalID);

            // Steps for the upwards stage (private member functions)
            coarsenUpwards_(tau, alpha, beta, gamma, omega);
            createIndexSets_(tau, alpha, beta, gamma, omega);
            createMatrixBlocks_(tau, alpha, beta, gamma, omega);
            mergeX_(tau, alpha, beta, gamma, omega);
            mergeW_(tau, alpha, beta, gamma, omega);
            mergeH_(tau, alpha, beta, gamma, omega);
            reorderOperatorsUpwards_(tau, alpha, beta, gamma, omega);

            // coarsenUpwards(tau, alpha, beta, gamma, omega);
            // createIndexSets(tau, alpha, beta, gamma, omega);
            // createMatrixBlocks(tau, alpha, beta, gamma, omega);
            // mergeX(tau, alpha, beta, gamma, omega);
            // mergeW(tau, alpha, beta, gamma, omega);
            // mergeH(tau, alpha, beta, gamma, omega);
            // reorderOperatorsUpwards(tau, alpha, beta, gamma, omega);
        }

        return;
    }

    /**
     * @brief Recursive split function
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    virtual void split1to4(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        // app.logHead("Splitting:");
        // app.logHead("  tau = %i", tau.globalID);
        // app.logHead("  alpha = %i", alpha.globalID);
        // app.logHead("  beta = %i", beta.globalID);
        // app.logHead("  gamma = %i", gamma.globalID);
        // app.logHead("  omega = %i", omega.globalID);

        // Steps for the split (private member functions)
        uncoarsen_(tau, alpha, beta, gamma, omega);
        applyS_(tau, alpha, beta, gamma, omega);
        // uncoarsen(tau, alpha, beta, gamma, omega);
        // applyS(tau, alpha, beta, gamma, omega);

        return;
    }

    /**
     * @brief Leaf solve function wrapper
     * 
     * @param patch Leaf patch
     */
    virtual void leafSolve(PatchType& patch) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (std::get<bool>(app.options["homogeneous-rhs"])) {
            // Need to set RHS to zeros for patch patch_solver b/c it won't be set already
            patch.vectorF() = Vector<double>(patch.grid().nx() * patch.grid().ny(), 0);
        }

        // Compute interior solution data
        patch.vectorU() = patch_solver.solve(patch.grid(), patch.vectorG(), patch.vectorF());

        return;
    }

private:

    // Index sets
    Vector<int> IS_alpha_beta_;
    Vector<int> IS_alpha_gamma_;
    Vector<int> IS_alpha_omega_;
    Vector<int> IS_alpha_tau_;

    Vector<int> IS_beta_alpha_;
    Vector<int> IS_beta_gamma_;
    Vector<int> IS_beta_omega_;
    Vector<int> IS_beta_tau_;

    Vector<int> IS_gamma_alpha_;
    Vector<int> IS_gamma_beta_;
    Vector<int> IS_gamma_omega_;
    Vector<int> IS_gamma_tau_;

    Vector<int> IS_omega_alpha_;
    Vector<int> IS_omega_beta_;
    Vector<int> IS_omega_gamma_;
    Vector<int> IS_omega_tau_;

    // Blocks for X_tau
    Matrix<double> T_ag_ag;
    Matrix<double> T_ga_ga;
    Matrix<double> T_ag_ab;
    Matrix<double> T_ga_go;
    Matrix<double> T_bo_bo;
    Matrix<double> T_ob_ob;
    Matrix<double> T_bo_ba;
    Matrix<double> T_ob_og;
    Matrix<double> T_ab_ag;
    Matrix<double> T_ba_bo;
    Matrix<double> T_ab_ab;
    Matrix<double> T_ba_ba;
    Matrix<double> T_go_ga;
    Matrix<double> T_og_ob;
    Matrix<double> T_go_go;
    Matrix<double> T_og_og;

    // Blocks for S_tau
    Matrix<double> T_ag_at;
    Matrix<double> T_ga_gt;
    Matrix<double> T_bo_bt;
    Matrix<double> T_ob_ot;
    Matrix<double> T_ab_at;
    Matrix<double> T_ba_bt;
    Matrix<double> T_go_gt;
    Matrix<double> T_og_ot;

    // Blocks for T_tau
    Matrix<double> T_at_at;
    Matrix<double> T_bt_bt;
    Matrix<double> T_gt_gt;
    Matrix<double> T_ot_ot;
    Matrix<double> T_at_ag;
    Matrix<double> T_at_ab;
    Matrix<double> T_bt_bo;
    Matrix<double> T_bt_ba;
    Matrix<double> T_gt_ga;
    Matrix<double> T_gt_go;
    Matrix<double> T_ot_ob;
    Matrix<double> T_ot_og;

    // ====================================================================================================
    // Steps for the merge
    // ====================================================================================================
    /**
     * @brief Tags patches for coarsening based on 
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     * @return Vector<int> 
     */
    Vector<int> tagPatchesForCoarsening_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        // Get data
        std::vector<PatchType*> patches = {&alpha, &beta, &gamma, &omega};
        std::vector<PatchGridType*> grids = {&alpha.grid(), &beta.grid(), &gamma.grid(), &omega.grid()};
        std::vector<int> sides(4);
        std::vector<int> tags(4);
        
        // Get vector of side lengths
        for (auto i = 0; i < 4; i++) 
            sides[i] = patches[i]->size();

        // Get minimum side length
        int min_side = *std::min_element(sides.begin(), sides.end());

        // Get tags based on side lengths
        for (auto i = 0; i < 4; i++) 
            tags[i] = static_cast<int>(log2(sides[i])) - static_cast<int>(log2(min_side));

        return {tags};
    }

public:

    static Vector<int> tagPatchesForCoarsening(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        // Get data
        std::vector<PatchType*> patches = {&alpha, &beta, &gamma, &omega};
        std::vector<PatchGridType*> grids = {&alpha.grid(), &beta.grid(), &gamma.grid(), &omega.grid()};
        std::vector<int> sides(4);
        std::vector<int> tags(4);
        
        // Get vector of side lengths
        for (auto i = 0; i < 4; i++) 
            sides[i] = patches[i]->size();

        // Get minimum side length
        int min_side = *std::min_element(sides.begin(), sides.end());

        // Get tags based on side lengths
        for (auto i = 0; i < 4; i++) 
            tags[i] = static_cast<int>(log2(sides[i])) - static_cast<int>(log2(min_side));

        return {tags};
    }

private:

    /**
     * @brief Coarsens the requisite data for the merge algorithm (DtN matrix)
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void coarsen_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patch_pointers = {&alpha, &beta, &gamma, &omega};
        Vector<int> tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);

        // Iterate over patches
        for (auto i = 0; i < 4; i++) {
            auto& patch = *patch_pointers[i];

            // Iterate over number of tagged
            for (auto n = 0; n < tags[i]; n++) {
                auto& grid = patch.grid();
                int ngrid = grid.nx();
                int coarsen_factor = tags[i] - (tags[i] - n);
                int nfine = ngrid / pow(2, coarsen_factor);
                int ncoarse = nfine / 2;

                InterpolationMatrixFine2Coarse<double> L21_side(ncoarse);
                std::vector<Matrix<double>> L21_diagonals = {L21_side, L21_side, L21_side, L21_side};
                Matrix<double> L21_patch = blockDiagonalMatrix(L21_diagonals);

                InterpolationMatrixCoarse2Fine<double> L12_side(nfine);
                std::vector<Matrix<double>> L12_diagonals = {L12_side, L12_side, L12_side, L12_side};
                Matrix<double> L12_patch = blockDiagonalMatrix(L12_diagonals);

                patch.matrixT() = L21_patch * patch.matrixT();
                patch.matrixT() = patch.matrixT() * L12_patch;

                patch.n_coarsens++;
            }
        }

        return;
    }

public:

    static void coarsen(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patch_pointers = {&alpha, &beta, &gamma, &omega};
        Vector<int> tags = tagPatchesForCoarsening(tau, alpha, beta, gamma, omega);

        // Iterate over patches
        for (auto i = 0; i < 4; i++) {
            auto& patch = *patch_pointers[i];

            // Iterate over number of tagged
            for (auto n = 0; n < tags[i]; n++) {
                auto& grid = patch.grid();
                int ngrid = grid.nx();
                int coarsen_factor = tags[i] - (tags[i] - n);
                int nfine = ngrid / pow(2, coarsen_factor);
                int ncoarse = nfine / 2;

                InterpolationMatrixFine2Coarse<double> L21_side(ncoarse);
                std::vector<Matrix<double>> L21_diagonals = {L21_side, L21_side, L21_side, L21_side};
                Matrix<double> L21_patch = blockDiagonalMatrix(L21_diagonals);

                InterpolationMatrixCoarse2Fine<double> L12_side(nfine);
                std::vector<Matrix<double>> L12_diagonals = {L12_side, L12_side, L12_side, L12_side};
                Matrix<double> L12_patch = blockDiagonalMatrix(L12_diagonals);

                patch.matrixT() = L21_patch * patch.matrixT();
                patch.matrixT() = patch.matrixT() * L12_patch;

                patch.n_coarsens++;
            }
        }

        return;
    }

private:

    /**
     * @brief Creates the index sets for the HPS method
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void createIndexSets_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check that all children patches are the same size (should be handled from the coarsening step if not)
        int nalpha = alpha.size();
        int nbeta = beta.size();
        int ngamma = gamma.size();
        int nomega = omega.size();
        Vector<int> n = {nalpha, nbeta, ngamma, nomega};
        if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
            throw std::invalid_argument("[EllipticForest::FISHPACK::FISHPACKHPSMethod::createIndexSets_] Size of children patches are not the same; something probably went wrong with the coarsening...");
        }

        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        IS_alpha_beta_ = I_E;
        IS_alpha_gamma_ = I_N;
        IS_alpha_tau_ = concatenate({I_W, I_S});
        
        IS_beta_alpha_ = I_W;
        IS_beta_omega_ = I_N;
        IS_beta_tau_ = concatenate({I_E, I_S});
        
        IS_gamma_alpha_ = I_S;
        IS_gamma_omega_ = I_E;
        IS_gamma_tau_ = concatenate({I_W, I_N});

        IS_omega_beta_ = I_S;
        IS_omega_gamma_ = I_W;
        IS_omega_tau_ = concatenate({I_E, I_N});

        return;
    }

    /**
     * @brief Creates the matrix blocks needed for the HPS method
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void createMatrixBlocks_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        Matrix<double>& T_alpha = alpha.matrixT();
        Matrix<double>& T_beta = beta.matrixT();
        Matrix<double>& T_gamma = gamma.matrixT();
        Matrix<double>& T_omega = omega.matrixT();

        // Blocks for X_tau
        T_ag_ag = T_alpha(IS_alpha_gamma_, IS_alpha_gamma_);
        T_ga_ga = T_gamma(IS_gamma_alpha_, IS_gamma_alpha_);
        T_ag_ab = T_alpha(IS_alpha_gamma_, IS_alpha_beta_);
        T_ga_go = T_gamma(IS_gamma_alpha_, IS_gamma_omega_);
        T_bo_bo = T_beta(IS_beta_omega_, IS_beta_omega_);
        T_ob_ob = T_omega(IS_omega_beta_, IS_omega_beta_);
        T_bo_ba = T_beta(IS_beta_omega_, IS_beta_alpha_);
        T_ob_og = T_omega(IS_omega_beta_, IS_omega_gamma_);
        T_ab_ag = T_alpha(IS_alpha_beta_, IS_alpha_gamma_);
        T_ba_bo = T_beta(IS_beta_alpha_, IS_beta_omega_);
        T_ab_ab = T_alpha(IS_alpha_beta_, IS_alpha_beta_);
        T_ba_ba = T_beta(IS_beta_alpha_, IS_beta_alpha_);
        T_go_ga = T_gamma(IS_gamma_omega_, IS_gamma_alpha_);
        T_og_ob = T_omega(IS_omega_gamma_, IS_omega_beta_);
        T_go_go = T_gamma(IS_gamma_omega_, IS_gamma_omega_);
        T_og_og = T_omega(IS_omega_gamma_, IS_omega_gamma_);

        // Blocks for S_tau
        T_ag_at = T_alpha(IS_alpha_gamma_, IS_alpha_tau_);
        T_ga_gt = T_gamma(IS_gamma_alpha_, IS_gamma_tau_);
        T_bo_bt = T_beta(IS_beta_omega_, IS_beta_tau_);
        T_ob_ot = T_omega(IS_omega_beta_, IS_omega_tau_);
        T_ab_at = T_alpha(IS_alpha_beta_, IS_alpha_tau_);
        T_ba_bt = T_beta(IS_beta_alpha_, IS_beta_tau_);
        T_go_gt = T_gamma(IS_gamma_omega_, IS_gamma_tau_);
        T_og_ot = T_omega(IS_omega_gamma_, IS_omega_tau_);

        // Blocks for T_tau
        T_at_at = T_alpha(IS_alpha_tau_, IS_alpha_tau_);
        T_bt_bt = T_beta(IS_beta_tau_, IS_beta_tau_);
        T_gt_gt = T_gamma(IS_gamma_tau_, IS_gamma_tau_);
        T_ot_ot = T_omega(IS_omega_tau_, IS_omega_tau_);
        T_at_ag = T_alpha(IS_alpha_tau_, IS_alpha_gamma_);
        T_at_ab = T_alpha(IS_alpha_tau_, IS_alpha_beta_);
        T_bt_bo = T_beta(IS_beta_tau_, IS_beta_omega_);
        T_bt_ba = T_beta(IS_beta_tau_, IS_beta_alpha_);
        T_gt_ga = T_gamma(IS_gamma_tau_, IS_gamma_alpha_);
        T_gt_go = T_gamma(IS_gamma_tau_, IS_gamma_omega_);
        T_ot_ob = T_omega(IS_omega_tau_, IS_omega_beta_);
        T_ot_og = T_omega(IS_omega_tau_, IS_omega_gamma_);

        // Negate blocks that need it
        T_ga_go = -T_ga_go;
        T_ob_og = -T_ob_og;
        T_ba_bo = -T_ba_bo;
        T_og_ob = -T_og_ob;
        T_ag_at = -T_ag_at;
        T_bo_bt = -T_bo_bt;
        T_ab_at = -T_ab_at;
        T_go_gt = -T_go_gt;

        return;
    }

    /**
     * @brief Computes the merged matrix X
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergeX_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create diagonals
        Matrix<double> T_diag1 = T_ag_ag - T_ga_ga;
        Matrix<double> T_diag2 = T_bo_bo - T_ob_ob;
        Matrix<double> T_diag3 = T_ab_ab - T_ba_ba;
        Matrix<double> T_diag4 = T_go_go - T_og_og;
        std::vector<Matrix<double>> diag = {T_diag1, T_diag2, T_diag3, T_diag4};

        // Create row and column block index starts
        std::vector<std::size_t> row_starts = { 0, T_diag1.nRows(), T_diag1.nRows() + T_diag2.nRows(), T_diag1.nRows() + T_diag2.nRows() + T_diag3.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_diag1.nCols(), T_diag1.nCols() + T_diag2.nCols(), T_diag1.nCols() + T_diag2.nCols() + T_diag3.nCols() };
        
        // Create matrix and set blocks
        tau.matrixX() = blockDiagonalMatrix(diag);
        tau.matrixX().setBlock(row_starts[0], col_starts[2], T_ag_ab);
        tau.matrixX().setBlock(row_starts[0], col_starts[3], T_ga_go);
        tau.matrixX().setBlock(row_starts[1], col_starts[2], T_bo_ba);
        tau.matrixX().setBlock(row_starts[1], col_starts[3], T_ob_og);
        tau.matrixX().setBlock(row_starts[2], col_starts[0], T_ab_ag);
        tau.matrixX().setBlock(row_starts[2], col_starts[1], T_ba_bo);
        tau.matrixX().setBlock(row_starts[3], col_starts[0], T_go_ga);
        tau.matrixX().setBlock(row_starts[3], col_starts[1], T_og_ob);

        return;
    }

public:

    static void mergeX(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create index sets
        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        Vector<int> IS_alpha_beta_ = I_E;
        Vector<int> IS_alpha_gamma_ = I_N;
        Vector<int> IS_alpha_tau_ = concatenate({I_W, I_S});
        
        Vector<int> IS_beta_alpha_ = I_W;
        Vector<int> IS_beta_omega_ = I_N;
        Vector<int> IS_beta_tau_ = concatenate({I_E, I_S});
        
        Vector<int> IS_gamma_alpha_ = I_S;
        Vector<int> IS_gamma_omega_ = I_E;
        Vector<int> IS_gamma_tau_ = concatenate({I_W, I_N});

        Vector<int> IS_omega_beta_ = I_S;
        Vector<int> IS_omega_gamma_ = I_W;
        Vector<int> IS_omega_tau_ = concatenate({I_E, I_N});

        // Create sub-matrices
        // Blocks for X_tau
        Matrix<NumericalType> T_ag_ag = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ga_ga = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_alpha_);
        Matrix<NumericalType> T_ag_ab = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_beta_);
        Matrix<NumericalType> T_ga_go = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_omega_);
        Matrix<NumericalType> T_bo_bo = beta.matrixT()(IS_beta_omega_, IS_beta_omega_);
        Matrix<NumericalType> T_ob_ob = omega.matrixT()(IS_omega_beta_, IS_omega_beta_);
        Matrix<NumericalType> T_bo_ba = beta.matrixT()(IS_beta_omega_, IS_beta_alpha_);
        Matrix<NumericalType> T_ob_og = omega.matrixT()(IS_omega_beta_, IS_omega_gamma_);
        Matrix<NumericalType> T_ab_ag = alpha.matrixT()(IS_alpha_beta_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ba_bo = beta.matrixT()(IS_beta_alpha_, IS_beta_omega_);
        Matrix<NumericalType> T_ab_ab = alpha.matrixT()(IS_alpha_beta_, IS_alpha_beta_);
        Matrix<NumericalType> T_ba_ba = beta.matrixT()(IS_beta_alpha_, IS_beta_alpha_);
        Matrix<NumericalType> T_go_ga = gamma.matrixT()(IS_gamma_omega_, IS_gamma_alpha_);
        Matrix<NumericalType> T_og_ob = omega.matrixT()(IS_omega_gamma_, IS_omega_beta_);
        Matrix<NumericalType> T_go_go = gamma.matrixT()(IS_gamma_omega_, IS_gamma_omega_);
        Matrix<NumericalType> T_og_og = omega.matrixT()(IS_omega_gamma_, IS_omega_gamma_);

        // Blocks for S_tau
        Matrix<NumericalType> T_ag_at = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_tau_);
        Matrix<NumericalType> T_ga_gt = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_tau_);
        Matrix<NumericalType> T_bo_bt = beta.matrixT()(IS_beta_omega_, IS_beta_tau_);
        Matrix<NumericalType> T_ob_ot = omega.matrixT()(IS_omega_beta_, IS_omega_tau_);
        Matrix<NumericalType> T_ab_at = alpha.matrixT()(IS_alpha_beta_, IS_alpha_tau_);
        Matrix<NumericalType> T_ba_bt = beta.matrixT()(IS_beta_alpha_, IS_beta_tau_);
        Matrix<NumericalType> T_go_gt = gamma.matrixT()(IS_gamma_omega_, IS_gamma_tau_);
        Matrix<NumericalType> T_og_ot = omega.matrixT()(IS_omega_gamma_, IS_omega_tau_);

        // Blocks for T_tau
        Matrix<NumericalType> T_at_at = alpha.matrixT()(IS_alpha_tau_, IS_alpha_tau_);
        Matrix<NumericalType> T_bt_bt = beta.matrixT()(IS_beta_tau_, IS_beta_tau_);
        Matrix<NumericalType> T_gt_gt = gamma.matrixT()(IS_gamma_tau_, IS_gamma_tau_);
        Matrix<NumericalType> T_ot_ot = omega.matrixT()(IS_omega_tau_, IS_omega_tau_);
        Matrix<NumericalType> T_at_ag = alpha.matrixT()(IS_alpha_tau_, IS_alpha_gamma_);
        Matrix<NumericalType> T_at_ab = alpha.matrixT()(IS_alpha_tau_, IS_alpha_beta_);
        Matrix<NumericalType> T_bt_bo = beta.matrixT()(IS_beta_tau_, IS_beta_omega_);
        Matrix<NumericalType> T_bt_ba = beta.matrixT()(IS_beta_tau_, IS_beta_alpha_);
        Matrix<NumericalType> T_gt_ga = gamma.matrixT()(IS_gamma_tau_, IS_gamma_alpha_);
        Matrix<NumericalType> T_gt_go = gamma.matrixT()(IS_gamma_tau_, IS_gamma_omega_);
        Matrix<NumericalType> T_ot_ob = omega.matrixT()(IS_omega_tau_, IS_omega_beta_);
        Matrix<NumericalType> T_ot_og = omega.matrixT()(IS_omega_tau_, IS_omega_gamma_);

        // Negate blocks that need it
        T_ga_go = -T_ga_go;
        T_ob_og = -T_ob_og;
        T_ba_bo = -T_ba_bo;
        T_og_ob = -T_og_ob;
        T_ag_at = -T_ag_at;
        T_bo_bt = -T_bo_bt;
        T_ab_at = -T_ab_at;
        T_go_gt = -T_go_gt;

        // Create diagonals
        Matrix<double> T_diag1 = T_ag_ag - T_ga_ga;
        Matrix<double> T_diag2 = T_bo_bo - T_ob_ob;
        Matrix<double> T_diag3 = T_ab_ab - T_ba_ba;
        Matrix<double> T_diag4 = T_go_go - T_og_og;
        std::vector<Matrix<double>> diag = {T_diag1, T_diag2, T_diag3, T_diag4};

        // Create row and column block index starts
        std::vector<std::size_t> row_starts = { 0, T_diag1.nRows(), T_diag1.nRows() + T_diag2.nRows(), T_diag1.nRows() + T_diag2.nRows() + T_diag3.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_diag1.nCols(), T_diag1.nCols() + T_diag2.nCols(), T_diag1.nCols() + T_diag2.nCols() + T_diag3.nCols() };
        
        // Create matrix and set blocks
        tau.matrixX() = blockDiagonalMatrix(diag);
        tau.matrixX().setBlock(row_starts[0], col_starts[2], T_ag_ab);
        tau.matrixX().setBlock(row_starts[0], col_starts[3], T_ga_go);
        tau.matrixX().setBlock(row_starts[1], col_starts[2], T_bo_ba);
        tau.matrixX().setBlock(row_starts[1], col_starts[3], T_ob_og);
        tau.matrixX().setBlock(row_starts[2], col_starts[0], T_ab_ag);
        tau.matrixX().setBlock(row_starts[2], col_starts[1], T_ba_bo);
        tau.matrixX().setBlock(row_starts[3], col_starts[0], T_go_ga);
        tau.matrixX().setBlock(row_starts[3], col_starts[1], T_og_ob);

    }

private:

    /**
     * @brief Computes the merged matrix S
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergeS_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create right hand side
        std::size_t nrows = T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() + T_go_gt.nRows();
        std::size_t ncols = T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() + T_ob_ot.nCols();
        Matrix<double> S_RHS(nrows, ncols, 0);

        std::vector<std::size_t> row_starts = { 0, T_ag_at.nRows(), T_ag_at.nRows() + T_bo_bt.nRows(), T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_ag_at.nCols(), T_ag_at.nCols() + T_bo_bt.nCols(), T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() };

        S_RHS.setBlock(row_starts[0], col_starts[0], T_ag_at);
        S_RHS.setBlock(row_starts[0], col_starts[2], T_ga_gt);
        S_RHS.setBlock(row_starts[1], col_starts[1], T_bo_bt);
        S_RHS.setBlock(row_starts[1], col_starts[3], T_ob_ot);
        S_RHS.setBlock(row_starts[2], col_starts[0], T_ab_at);
        S_RHS.setBlock(row_starts[2], col_starts[1], T_ba_bt);
        S_RHS.setBlock(row_starts[3], col_starts[2], T_go_gt);
        S_RHS.setBlock(row_starts[3], col_starts[3], T_og_ot);
        
        // Solve to set S_tau
        tau.matrixS() = solve(tau.matrixX(), S_RHS);

        return;
    }

public:

    static void mergeS(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create index sets
        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        Vector<int> IS_alpha_beta_ = I_E;
        Vector<int> IS_alpha_gamma_ = I_N;
        Vector<int> IS_alpha_tau_ = concatenate({I_W, I_S});
        
        Vector<int> IS_beta_alpha_ = I_W;
        Vector<int> IS_beta_omega_ = I_N;
        Vector<int> IS_beta_tau_ = concatenate({I_E, I_S});
        
        Vector<int> IS_gamma_alpha_ = I_S;
        Vector<int> IS_gamma_omega_ = I_E;
        Vector<int> IS_gamma_tau_ = concatenate({I_W, I_N});

        Vector<int> IS_omega_beta_ = I_S;
        Vector<int> IS_omega_gamma_ = I_W;
        Vector<int> IS_omega_tau_ = concatenate({I_E, I_N});

        // Create sub-matrices
        // Blocks for X_tau
        Matrix<NumericalType> T_ag_ag = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ga_ga = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_alpha_);
        Matrix<NumericalType> T_ag_ab = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_beta_);
        Matrix<NumericalType> T_ga_go = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_omega_);
        Matrix<NumericalType> T_bo_bo = beta.matrixT()(IS_beta_omega_, IS_beta_omega_);
        Matrix<NumericalType> T_ob_ob = omega.matrixT()(IS_omega_beta_, IS_omega_beta_);
        Matrix<NumericalType> T_bo_ba = beta.matrixT()(IS_beta_omega_, IS_beta_alpha_);
        Matrix<NumericalType> T_ob_og = omega.matrixT()(IS_omega_beta_, IS_omega_gamma_);
        Matrix<NumericalType> T_ab_ag = alpha.matrixT()(IS_alpha_beta_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ba_bo = beta.matrixT()(IS_beta_alpha_, IS_beta_omega_);
        Matrix<NumericalType> T_ab_ab = alpha.matrixT()(IS_alpha_beta_, IS_alpha_beta_);
        Matrix<NumericalType> T_ba_ba = beta.matrixT()(IS_beta_alpha_, IS_beta_alpha_);
        Matrix<NumericalType> T_go_ga = gamma.matrixT()(IS_gamma_omega_, IS_gamma_alpha_);
        Matrix<NumericalType> T_og_ob = omega.matrixT()(IS_omega_gamma_, IS_omega_beta_);
        Matrix<NumericalType> T_go_go = gamma.matrixT()(IS_gamma_omega_, IS_gamma_omega_);
        Matrix<NumericalType> T_og_og = omega.matrixT()(IS_omega_gamma_, IS_omega_gamma_);

        // Blocks for S_tau
        Matrix<NumericalType> T_ag_at = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_tau_);
        Matrix<NumericalType> T_ga_gt = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_tau_);
        Matrix<NumericalType> T_bo_bt = beta.matrixT()(IS_beta_omega_, IS_beta_tau_);
        Matrix<NumericalType> T_ob_ot = omega.matrixT()(IS_omega_beta_, IS_omega_tau_);
        Matrix<NumericalType> T_ab_at = alpha.matrixT()(IS_alpha_beta_, IS_alpha_tau_);
        Matrix<NumericalType> T_ba_bt = beta.matrixT()(IS_beta_alpha_, IS_beta_tau_);
        Matrix<NumericalType> T_go_gt = gamma.matrixT()(IS_gamma_omega_, IS_gamma_tau_);
        Matrix<NumericalType> T_og_ot = omega.matrixT()(IS_omega_gamma_, IS_omega_tau_);

        // Blocks for T_tau
        Matrix<NumericalType> T_at_at = alpha.matrixT()(IS_alpha_tau_, IS_alpha_tau_);
        Matrix<NumericalType> T_bt_bt = beta.matrixT()(IS_beta_tau_, IS_beta_tau_);
        Matrix<NumericalType> T_gt_gt = gamma.matrixT()(IS_gamma_tau_, IS_gamma_tau_);
        Matrix<NumericalType> T_ot_ot = omega.matrixT()(IS_omega_tau_, IS_omega_tau_);
        Matrix<NumericalType> T_at_ag = alpha.matrixT()(IS_alpha_tau_, IS_alpha_gamma_);
        Matrix<NumericalType> T_at_ab = alpha.matrixT()(IS_alpha_tau_, IS_alpha_beta_);
        Matrix<NumericalType> T_bt_bo = beta.matrixT()(IS_beta_tau_, IS_beta_omega_);
        Matrix<NumericalType> T_bt_ba = beta.matrixT()(IS_beta_tau_, IS_beta_alpha_);
        Matrix<NumericalType> T_gt_ga = gamma.matrixT()(IS_gamma_tau_, IS_gamma_alpha_);
        Matrix<NumericalType> T_gt_go = gamma.matrixT()(IS_gamma_tau_, IS_gamma_omega_);
        Matrix<NumericalType> T_ot_ob = omega.matrixT()(IS_omega_tau_, IS_omega_beta_);
        Matrix<NumericalType> T_ot_og = omega.matrixT()(IS_omega_tau_, IS_omega_gamma_);

        // Negate blocks that need it
        T_ga_go = -T_ga_go;
        T_ob_og = -T_ob_og;
        T_ba_bo = -T_ba_bo;
        T_og_ob = -T_og_ob;
        T_ag_at = -T_ag_at;
        T_bo_bt = -T_bo_bt;
        T_ab_at = -T_ab_at;
        T_go_gt = -T_go_gt;

        // Create right hand side
        std::size_t nrows = T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() + T_go_gt.nRows();
        std::size_t ncols = T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() + T_ob_ot.nCols();
        Matrix<double> S_RHS(nrows, ncols, 0);

        std::vector<std::size_t> row_starts = { 0, T_ag_at.nRows(), T_ag_at.nRows() + T_bo_bt.nRows(), T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_ag_at.nCols(), T_ag_at.nCols() + T_bo_bt.nCols(), T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() };

        S_RHS.setBlock(row_starts[0], col_starts[0], T_ag_at);
        S_RHS.setBlock(row_starts[0], col_starts[2], T_ga_gt);
        S_RHS.setBlock(row_starts[1], col_starts[1], T_bo_bt);
        S_RHS.setBlock(row_starts[1], col_starts[3], T_ob_ot);
        S_RHS.setBlock(row_starts[2], col_starts[0], T_ab_at);
        S_RHS.setBlock(row_starts[2], col_starts[1], T_ba_bt);
        S_RHS.setBlock(row_starts[3], col_starts[2], T_go_gt);
        S_RHS.setBlock(row_starts[3], col_starts[3], T_og_ot);
        
        // Solve to set S_tau
        tau.matrixS() = solve(tau.matrixX(), S_RHS);

        return;
    }

private:

    /**
     * @brief Computes the merged matrix T
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergeT_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create left hand side
        std::vector<Matrix<double>> diag = {T_at_at, T_bt_bt, T_gt_gt, T_ot_ot};
        Matrix<double> T_LHS = blockDiagonalMatrix(diag);

        // Create right hand side
        std::size_t nrows = T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() + T_ot_ob.nRows();
        std::size_t ncols = T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() + T_gt_go.nCols();
        tau.matrixH() = Matrix<double>(nrows, ncols, 0);

        std::vector<std::size_t> row_starts = { 0, T_at_ag.nRows(), T_at_ag.nRows() + T_bt_bo.nRows(), T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_at_ag.nCols(), T_at_ag.nCols() + T_bt_bo.nCols(), T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() };

        tau.matrixH().setBlock(row_starts[0], col_starts[0], T_at_ag);
        tau.matrixH().setBlock(row_starts[0], col_starts[2], T_at_ab);
        tau.matrixH().setBlock(row_starts[1], col_starts[1], T_bt_bo);
        tau.matrixH().setBlock(row_starts[1], col_starts[2], T_bt_ba);
        tau.matrixH().setBlock(row_starts[2], col_starts[0], T_gt_ga);
        tau.matrixH().setBlock(row_starts[2], col_starts[3], T_gt_go);
        tau.matrixH().setBlock(row_starts[3], col_starts[1], T_ot_ob);
        tau.matrixH().setBlock(row_starts[3], col_starts[3], T_ot_og);

        // Compute and set T_tau
        Matrix<double> T_RHS = tau.matrixH() * tau.matrixS();
        tau.matrixT() = T_LHS + T_RHS;

        return;
    }

public:

    static void mergeT(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create index sets
        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        Vector<int> IS_alpha_beta_ = I_E;
        Vector<int> IS_alpha_gamma_ = I_N;
        Vector<int> IS_alpha_tau_ = concatenate({I_W, I_S});
        
        Vector<int> IS_beta_alpha_ = I_W;
        Vector<int> IS_beta_omega_ = I_N;
        Vector<int> IS_beta_tau_ = concatenate({I_E, I_S});
        
        Vector<int> IS_gamma_alpha_ = I_S;
        Vector<int> IS_gamma_omega_ = I_E;
        Vector<int> IS_gamma_tau_ = concatenate({I_W, I_N});

        Vector<int> IS_omega_beta_ = I_S;
        Vector<int> IS_omega_gamma_ = I_W;
        Vector<int> IS_omega_tau_ = concatenate({I_E, I_N});

        // Create sub-matrices
        // Blocks for X_tau
        Matrix<NumericalType> T_ag_ag = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ga_ga = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_alpha_);
        Matrix<NumericalType> T_ag_ab = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_beta_);
        Matrix<NumericalType> T_ga_go = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_omega_);
        Matrix<NumericalType> T_bo_bo = beta.matrixT()(IS_beta_omega_, IS_beta_omega_);
        Matrix<NumericalType> T_ob_ob = omega.matrixT()(IS_omega_beta_, IS_omega_beta_);
        Matrix<NumericalType> T_bo_ba = beta.matrixT()(IS_beta_omega_, IS_beta_alpha_);
        Matrix<NumericalType> T_ob_og = omega.matrixT()(IS_omega_beta_, IS_omega_gamma_);
        Matrix<NumericalType> T_ab_ag = alpha.matrixT()(IS_alpha_beta_, IS_alpha_gamma_);
        Matrix<NumericalType> T_ba_bo = beta.matrixT()(IS_beta_alpha_, IS_beta_omega_);
        Matrix<NumericalType> T_ab_ab = alpha.matrixT()(IS_alpha_beta_, IS_alpha_beta_);
        Matrix<NumericalType> T_ba_ba = beta.matrixT()(IS_beta_alpha_, IS_beta_alpha_);
        Matrix<NumericalType> T_go_ga = gamma.matrixT()(IS_gamma_omega_, IS_gamma_alpha_);
        Matrix<NumericalType> T_og_ob = omega.matrixT()(IS_omega_gamma_, IS_omega_beta_);
        Matrix<NumericalType> T_go_go = gamma.matrixT()(IS_gamma_omega_, IS_gamma_omega_);
        Matrix<NumericalType> T_og_og = omega.matrixT()(IS_omega_gamma_, IS_omega_gamma_);

        // Blocks for S_tau
        Matrix<NumericalType> T_ag_at = alpha.matrixT()(IS_alpha_gamma_, IS_alpha_tau_);
        Matrix<NumericalType> T_ga_gt = gamma.matrixT()(IS_gamma_alpha_, IS_gamma_tau_);
        Matrix<NumericalType> T_bo_bt = beta.matrixT()(IS_beta_omega_, IS_beta_tau_);
        Matrix<NumericalType> T_ob_ot = omega.matrixT()(IS_omega_beta_, IS_omega_tau_);
        Matrix<NumericalType> T_ab_at = alpha.matrixT()(IS_alpha_beta_, IS_alpha_tau_);
        Matrix<NumericalType> T_ba_bt = beta.matrixT()(IS_beta_alpha_, IS_beta_tau_);
        Matrix<NumericalType> T_go_gt = gamma.matrixT()(IS_gamma_omega_, IS_gamma_tau_);
        Matrix<NumericalType> T_og_ot = omega.matrixT()(IS_omega_gamma_, IS_omega_tau_);

        // Blocks for T_tau
        Matrix<NumericalType> T_at_at = alpha.matrixT()(IS_alpha_tau_, IS_alpha_tau_);
        Matrix<NumericalType> T_bt_bt = beta.matrixT()(IS_beta_tau_, IS_beta_tau_);
        Matrix<NumericalType> T_gt_gt = gamma.matrixT()(IS_gamma_tau_, IS_gamma_tau_);
        Matrix<NumericalType> T_ot_ot = omega.matrixT()(IS_omega_tau_, IS_omega_tau_);
        Matrix<NumericalType> T_at_ag = alpha.matrixT()(IS_alpha_tau_, IS_alpha_gamma_);
        Matrix<NumericalType> T_at_ab = alpha.matrixT()(IS_alpha_tau_, IS_alpha_beta_);
        Matrix<NumericalType> T_bt_bo = beta.matrixT()(IS_beta_tau_, IS_beta_omega_);
        Matrix<NumericalType> T_bt_ba = beta.matrixT()(IS_beta_tau_, IS_beta_alpha_);
        Matrix<NumericalType> T_gt_ga = gamma.matrixT()(IS_gamma_tau_, IS_gamma_alpha_);
        Matrix<NumericalType> T_gt_go = gamma.matrixT()(IS_gamma_tau_, IS_gamma_omega_);
        Matrix<NumericalType> T_ot_ob = omega.matrixT()(IS_omega_tau_, IS_omega_beta_);
        Matrix<NumericalType> T_ot_og = omega.matrixT()(IS_omega_tau_, IS_omega_gamma_);

        // Negate blocks that need it
        T_ga_go = -T_ga_go;
        T_ob_og = -T_ob_og;
        T_ba_bo = -T_ba_bo;
        T_og_ob = -T_og_ob;
        T_ag_at = -T_ag_at;
        T_bo_bt = -T_bo_bt;
        T_ab_at = -T_ab_at;
        T_go_gt = -T_go_gt;

        // Create left hand side
        std::vector<Matrix<double>> diag = {T_at_at, T_bt_bt, T_gt_gt, T_ot_ot};
        Matrix<double> T_LHS = blockDiagonalMatrix(diag);

        // Create right hand side
        std::size_t nrows = T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() + T_ot_ob.nRows();
        std::size_t ncols = T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() + T_gt_go.nCols();
        tau.matrixH() = Matrix<double>(nrows, ncols, 0);

        std::vector<std::size_t> row_starts = { 0, T_at_ag.nRows(), T_at_ag.nRows() + T_bt_bo.nRows(), T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() };
        std::vector<std::size_t> col_starts = { 0, T_at_ag.nCols(), T_at_ag.nCols() + T_bt_bo.nCols(), T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() };

        tau.matrixH().setBlock(row_starts[0], col_starts[0], T_at_ag);
        tau.matrixH().setBlock(row_starts[0], col_starts[2], T_at_ab);
        tau.matrixH().setBlock(row_starts[1], col_starts[1], T_bt_bo);
        tau.matrixH().setBlock(row_starts[1], col_starts[2], T_bt_ba);
        tau.matrixH().setBlock(row_starts[2], col_starts[0], T_gt_ga);
        tau.matrixH().setBlock(row_starts[2], col_starts[3], T_gt_go);
        tau.matrixH().setBlock(row_starts[3], col_starts[1], T_ot_ob);
        tau.matrixH().setBlock(row_starts[3], col_starts[3], T_ot_og);

        // Compute and set T_tau
        Matrix<double> T_RHS = tau.matrixH() * tau.matrixS();
        tau.matrixT() = T_LHS + T_RHS;

        return;
    }

private:

    /**
     * @brief Performs a permutation on the matrices S and T to stay in WESN ordering
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void reorderOperators_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Form permutation vector and block sizes
        int nside = alpha.matrixT().nRows() / 4;
        Vector<int> pi_noChange = {0, 1, 2, 3};
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes1(4, nside);
        Vector<int> blockSizes2(8, nside);

        // Permute S and T
        tau.matrixS() = tau.matrixS().blockPermute(pi_noChange, pi_WESN, blockSizes1, blockSizes2);
        tau.matrixT() = tau.matrixT().blockPermute(pi_WESN, pi_WESN, blockSizes2, blockSizes2);

        return;
    }

public:

    static void reorderOperators(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Form permutation vector and block sizes
        int nside = alpha.matrixT().nRows() / 4;
        Vector<int> pi_noChange = {0, 1, 2, 3};
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes1(4, nside);
        Vector<int> blockSizes2(8, nside);

        // Permute S and T
        tau.matrixS() = tau.matrixS().blockPermute(pi_noChange, pi_WESN, blockSizes1, blockSizes2);
        tau.matrixT() = tau.matrixT().blockPermute(pi_WESN, pi_WESN, blockSizes2, blockSizes2);

        return;
    }

private:

    /**
     * @brief Merges the remaining data for the parent
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergePatch_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        PatchGridType merged_grid(MPI_COMM_SELF, alpha.size() + beta.size(), alpha.grid().xLower(), beta.grid().xUpper(), alpha.size() + gamma.size(), alpha.grid().yLower(), gamma.grid().yUpper());
        tau.grid() = merged_grid;

    }

public:

    static void mergePatch(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        PatchGridType merged_grid(MPI_COMM_SELF, alpha.size() + beta.size(), alpha.grid().xLower(), beta.grid().xUpper(), alpha.size() + gamma.size(), alpha.grid().yLower(), gamma.grid().yUpper());
        tau.grid() = merged_grid;

    }

private:

    // ====================================================================================================
    // Steps for the upwards stage
    // ====================================================================================================

    /**
     * @brief Coarsens the requisite data for the merge algorithm (h vector)
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void coarsenUpwards_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patch_pointers = {&alpha, &beta, &gamma, &omega};

        // Iterate over patches
        for (auto i = 0; i < 4; i++) {
            auto& patch = *patch_pointers[i];

            // Iterate over tagged patches
            for (auto n = 0; n < patch.n_coarsens; n++) {
                auto& grid = patch.grid();
                int ngrid = grid.nx();
                int nfine = ngrid / pow(2, n);
                int ncoarse = nfine / 2;

                InterpolationMatrixFine2Coarse<double> L21_side(ncoarse);
                std::vector<Matrix<double>> L21_diagonals = {L21_side, L21_side, L21_side, L21_side};
                Matrix<double> L21_patch = blockDiagonalMatrix(L21_diagonals);
                patch_pointers[i]->vectorH() = L21_patch * patch_pointers[i]->vectorH();
            }
        }

        return;
    }

public:

    /**
     * @brief Coarsens the requisite data for the merge algorithm (h vector)
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void coarsenUpwards(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patch_pointers = {&alpha, &beta, &gamma, &omega};

        // Iterate over patches
        for (auto i = 0; i < 4; i++) {
            auto& patch = *patch_pointers[i];

            // Iterate over tagged patches
            for (auto n = 0; n < patch.n_coarsens; n++) {
                auto& grid = patch.grid();
                int ngrid = grid.nx();
                int nfine = ngrid / pow(2, n);
                int ncoarse = nfine / 2;

                InterpolationMatrixFine2Coarse<double> L21_side(ncoarse);
                std::vector<Matrix<double>> L21_diagonals = {L21_side, L21_side, L21_side, L21_side};
                Matrix<double> L21_patch = blockDiagonalMatrix(L21_diagonals);
                patch_pointers[i]->vectorH() = L21_patch * patch_pointers[i]->vectorH();
            }
        }

        return;
    }

private:

    /**
     * @brief Computes the merged vector w
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergeW_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Form hDiff
        Vector<double> h_ga = gamma.vectorH()(IS_gamma_alpha_);
        Vector<double> h_ag = alpha.vectorH()(IS_alpha_gamma_);
        Vector<double> h_ob = omega.vectorH()(IS_omega_beta_);
        Vector<double> h_bo = beta.vectorH()(IS_beta_omega_);
        Vector<double> h_ba = beta.vectorH()(IS_beta_alpha_);
        Vector<double> h_ab = alpha.vectorH()(IS_alpha_beta_);
        Vector<double> h_og = omega.vectorH()(IS_omega_gamma_);
        Vector<double> h_go = gamma.vectorH()(IS_gamma_omega_);

        Vector<double> h_diff_gamma_alpha = h_ga - h_ag;
        Vector<double> h_diff_omega_beta = h_ob - h_bo;
        Vector<double> h_diff_beta_alpha = h_ba - h_ab;
        Vector<double> h_diff_omega_gamma = h_og - h_go;

        Vector<double> hDiff = concatenate({
            h_diff_gamma_alpha,
            h_diff_omega_beta,
            h_diff_beta_alpha,
            h_diff_omega_gamma
        });

        // Compute and set w_tau
        tau.vectorW() = solve(tau.matrixX(), hDiff);

        return;
    }

public:

    /**
     * @brief Computes the merged vector w
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void mergeW(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create index sets
        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        Vector<int> IS_alpha_beta_ = I_E;
        Vector<int> IS_alpha_gamma_ = I_N;
        Vector<int> IS_alpha_tau_ = concatenate({I_W, I_S});
        
        Vector<int> IS_beta_alpha_ = I_W;
        Vector<int> IS_beta_omega_ = I_N;
        Vector<int> IS_beta_tau_ = concatenate({I_E, I_S});
        
        Vector<int> IS_gamma_alpha_ = I_S;
        Vector<int> IS_gamma_omega_ = I_E;
        Vector<int> IS_gamma_tau_ = concatenate({I_W, I_N});

        Vector<int> IS_omega_beta_ = I_S;
        Vector<int> IS_omega_gamma_ = I_W;
        Vector<int> IS_omega_tau_ = concatenate({I_E, I_N});

        // Form hDiff
        Vector<double> h_ga = gamma.vectorH()(IS_gamma_alpha_);
        Vector<double> h_ag = alpha.vectorH()(IS_alpha_gamma_);
        Vector<double> h_ob = omega.vectorH()(IS_omega_beta_);
        Vector<double> h_bo = beta.vectorH()(IS_beta_omega_);
        Vector<double> h_ba = beta.vectorH()(IS_beta_alpha_);
        Vector<double> h_ab = alpha.vectorH()(IS_alpha_beta_);
        Vector<double> h_og = omega.vectorH()(IS_omega_gamma_);
        Vector<double> h_go = gamma.vectorH()(IS_gamma_omega_);

        Vector<double> h_diff_gamma_alpha = h_ga - h_ag;
        Vector<double> h_diff_omega_beta = h_ob - h_bo;
        Vector<double> h_diff_beta_alpha = h_ba - h_ab;
        Vector<double> h_diff_omega_gamma = h_og - h_go;

        Vector<double> hDiff = concatenate({
            h_diff_gamma_alpha,
            h_diff_omega_beta,
            h_diff_beta_alpha,
            h_diff_omega_gamma
        });

        // Compute and set w_tau
        tau.vectorW() = solve(tau.matrixX(), hDiff);

        return;
    }

private:

    /**
     * @brief Computes the merged vector h
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void mergeH_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Compute and set h_tau
        tau.vectorH() = tau.matrixH() * tau.vectorW();

        // Update with boundary h
        Vector<double> h_alpha_tau = alpha.vectorH()(IS_alpha_tau_);
        Vector<double> h_beta_tau = beta.vectorH()(IS_beta_tau_);
        Vector<double> h_gamma_tau = gamma.vectorH()(IS_gamma_tau_);
        Vector<double> h_omega_tau = omega.vectorH()(IS_omega_tau_);
        Vector<double> h_update = concatenate({
            h_alpha_tau,
            h_beta_tau,
            h_gamma_tau,
            h_omega_tau
        });
        tau.vectorH() += h_update;

        return;
    }

public:

    /**
     * @brief Computes the merged vector h
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void mergeH(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create index sets
        int nside = alpha.matrixT().nRows() / 4;

        Vector<int> I_W = vectorRange(0, nside-1);
        Vector<int> I_E = vectorRange(nside, 2*nside - 1);
        Vector<int> I_S = vectorRange(2*nside, 3*nside - 1);
        Vector<int> I_N = vectorRange(3*nside, 4*nside - 1);

        Vector<int> IS_alpha_beta_ = I_E;
        Vector<int> IS_alpha_gamma_ = I_N;
        Vector<int> IS_alpha_tau_ = concatenate({I_W, I_S});
        
        Vector<int> IS_beta_alpha_ = I_W;
        Vector<int> IS_beta_omega_ = I_N;
        Vector<int> IS_beta_tau_ = concatenate({I_E, I_S});
        
        Vector<int> IS_gamma_alpha_ = I_S;
        Vector<int> IS_gamma_omega_ = I_E;
        Vector<int> IS_gamma_tau_ = concatenate({I_W, I_N});

        Vector<int> IS_omega_beta_ = I_S;
        Vector<int> IS_omega_gamma_ = I_W;
        Vector<int> IS_omega_tau_ = concatenate({I_E, I_N});

        // Compute and set h_tau
        tau.vectorH() = tau.matrixH() * tau.vectorW();

        // Update with boundary h
        Vector<double> h_alpha_tau = alpha.vectorH()(IS_alpha_tau_);
        Vector<double> h_beta_tau = beta.vectorH()(IS_beta_tau_);
        Vector<double> h_gamma_tau = gamma.vectorH()(IS_gamma_tau_);
        Vector<double> h_omega_tau = omega.vectorH()(IS_omega_tau_);
        Vector<double> h_update = concatenate({
            h_alpha_tau,
            h_beta_tau,
            h_gamma_tau,
            h_omega_tau
        });
        tau.vectorH() += h_update;

        return;
    }

private:

    /**
     * @brief Performs a permutation on the vector h to stay in WESN ordering
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void reorderOperatorsUpwards_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // int nalpha = alpha.grid().nx();
        // int nbeta = beta.grid().nx();
        // int ngamma = gamma.grid().nx();
        // int nomega = omega.grid().nx();
        int nalpha = alpha.size();
        int nbeta = beta.size();
        int ngamma = gamma.size();
        int nomega = omega.size();
        Vector<int> n = {nalpha, nbeta, ngamma, nomega};
        if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
            throw std::invalid_argument("[EllipticForest::HPSAlgorithm::reorderOperatorsUpwards_] Size of children patches are not the same; something probably went wrong with the coarsening...");
        }

        // Form permutation vector and block sizes
        int nside = alpha.matrixT().nRows() / 4;
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes(8, nside);

        // Reorder
        tau.vectorH() = tau.vectorH().blockPermute(pi_WESN, blockSizes);
    }

public:

    /**
     * @brief Performs a permutation on the vector h to stay in WESN ordering
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void reorderOperatorsUpwards(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // int nalpha = alpha.grid().nx();
        // int nbeta = beta.grid().nx();
        // int ngamma = gamma.grid().nx();
        // int nomega = omega.grid().nx();
        int nalpha = alpha.size();
        int nbeta = beta.size();
        int ngamma = gamma.size();
        int nomega = omega.size();
        Vector<int> n = {nalpha, nbeta, ngamma, nomega};
        if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
            throw std::invalid_argument("[EllipticForest::HPSAlgorithm::reorderOperatorsUpwards_] Size of children patches are not the same; something probably went wrong with the coarsening...");
        }

        // Form permutation vector and block sizes
        int nside = alpha.matrixT().nRows() / 4;
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes(8, nside);

        // Reorder
        tau.vectorH() = tau.vectorH().blockPermute(pi_WESN, blockSizes);
    }

private:

    // ====================================================================================================
    // Steps for the solve stage
    // ====================================================================================================

    /**
     * @brief Uncoarsens a patch that was tagged for coarsening in the build stage by averaging data to finer data
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void uncoarsen_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        EllipticForestApp& app = EllipticForestApp::getInstance();

        for (auto n = 0; n < tau.n_coarsens; n++) {
            auto& grid = tau.grid();
            int ngrid = grid.nx();
            int coarsen_factor = tau.n_coarsens - (n + 1);
            int nfine = ngrid / pow(2, coarsen_factor);
            int ncoarse = nfine / 2;

            InterpolationMatrixCoarse2Fine<double> L12_side(nfine);
            std::vector<Matrix<double>> L12_diagonals = {L12_side, L12_side, L12_side, L12_side};
            Matrix<double> L12_patch = blockDiagonalMatrix(L12_diagonals);
            tau.vectorG() = L12_patch * tau.vectorG();
        }
     
        return;
    }

public:

    /**
     * @brief Uncoarsens a patch that was tagged for coarsening in the build stage by averaging data to finer data
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void uncoarsen(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        EllipticForestApp& app = EllipticForestApp::getInstance();

        for (auto n = 0; n < tau.n_coarsens; n++) {
            auto& grid = tau.grid();
            int ngrid = grid.nx();
            int coarsen_factor = tau.n_coarsens - (n + 1);
            int nfine = ngrid / pow(2, coarsen_factor);
            int ncoarse = nfine / 2;

            InterpolationMatrixCoarse2Fine<double> L12_side(nfine);
            std::vector<Matrix<double>> L12_diagonals = {L12_side, L12_side, L12_side, L12_side};
            Matrix<double> L12_patch = blockDiagonalMatrix(L12_diagonals);
            tau.vectorG() = L12_patch * tau.vectorG();
        }
     
        return;
    }

private:

    /**
     * @brief Applies the S operator to map the exterior solution data to interior solution data
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    void applyS_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Apply solution operator to get interior of tau
        Vector<double> u_tau_interior = tau.matrixS() * tau.vectorG();
        Vector<double> u_tau_interior_intermediate = u_tau_interior;

        // Apply non-homogeneous contribution
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            u_tau_interior = u_tau_interior + tau.vectorW();
        }

        // Extract components of interior of tau
        int nside = alpha.size();
        Vector<double> g_alpha_gamma = u_tau_interior.getSegment(0*nside, nside);
        Vector<double> g_beta_omega = u_tau_interior.getSegment(1*nside, nside);
        Vector<double> g_alpha_beta = u_tau_interior.getSegment(2*nside, nside);
        Vector<double> g_gamma_omega = u_tau_interior.getSegment(3*nside, nside);

        // Extract components of exterior of tau
        Vector<double> g_alpha_W = tau.vectorG().getSegment(0*nside, nside);
        Vector<double> g_gamma_W = tau.vectorG().getSegment(1*nside, nside);
        Vector<double> g_beta_E = tau.vectorG().getSegment(2*nside, nside);
        Vector<double> g_omega_E = tau.vectorG().getSegment(3*nside, nside);
        Vector<double> g_alpha_S = tau.vectorG().getSegment(4*nside, nside);
        Vector<double> g_beta_S = tau.vectorG().getSegment(5*nside, nside);
        Vector<double> g_gamma_N = tau.vectorG().getSegment(6*nside, nside);
        Vector<double> g_omega_N = tau.vectorG().getSegment(7*nside, nside);

        // Set child patch Dirichlet data
        alpha.vectorG() = concatenate({g_alpha_W, g_alpha_beta, g_alpha_S, g_alpha_gamma});
        beta.vectorG() = concatenate({g_alpha_beta, g_beta_E, g_beta_S, g_beta_omega});
        gamma.vectorG() = concatenate({g_gamma_W, g_gamma_omega, g_alpha_gamma, g_gamma_N});
        omega.vectorG() = concatenate({g_gamma_omega, g_omega_E, g_beta_omega, g_omega_N});

        return;
    }

public:

    /**
     * @brief Applies the S operator to map the exterior solution data to interior solution data
     * 
     * @param tau Parent patch
     * @param alpha Lower-left patch
     * @param beta Lower-right patch
     * @param gamma Upper-left patch
     * @param omega Upper-right patch
     */
    static void applyS(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Apply solution operator to get interior of tau
        Vector<double> u_tau_interior = tau.matrixS() * tau.vectorG();

        // Apply non-homogeneous contribution
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            u_tau_interior = u_tau_interior + tau.vectorW();
        }

        // Extract components of interior of tau
        int nside = alpha.size();
        Vector<double> g_alpha_gamma = u_tau_interior.getSegment(0*nside, nside);
        Vector<double> g_beta_omega = u_tau_interior.getSegment(1*nside, nside);
        Vector<double> g_alpha_beta = u_tau_interior.getSegment(2*nside, nside);
        Vector<double> g_gamma_omega = u_tau_interior.getSegment(3*nside, nside);

        // Extract components of exterior of tau
        Vector<double> g_alpha_W = tau.vectorG().getSegment(0*nside, nside);
        Vector<double> g_gamma_W = tau.vectorG().getSegment(1*nside, nside);
        Vector<double> g_beta_E = tau.vectorG().getSegment(2*nside, nside);
        Vector<double> g_omega_E = tau.vectorG().getSegment(3*nside, nside);
        Vector<double> g_alpha_S = tau.vectorG().getSegment(4*nside, nside);
        Vector<double> g_beta_S = tau.vectorG().getSegment(5*nside, nside);
        Vector<double> g_gamma_N = tau.vectorG().getSegment(6*nside, nside);
        Vector<double> g_omega_N = tau.vectorG().getSegment(7*nside, nside);

        // Set child patch Dirichlet data
        alpha.vectorG() = concatenate({g_alpha_W, g_alpha_beta, g_alpha_S, g_alpha_gamma});
        beta.vectorG() = concatenate({g_alpha_beta, g_beta_E, g_beta_S, g_beta_omega});
        gamma.vectorG() = concatenate({g_gamma_W, g_gamma_omega, g_alpha_gamma, g_gamma_N});
        omega.vectorG() = concatenate({g_gamma_omega, g_omega_E, g_beta_omega, g_omega_N});

        return;
    }

};

} // NAMESPACE : EllipticForest

#endif // HPS_ALGORITHM_HPP_