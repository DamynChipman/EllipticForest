#ifndef HPS_ALGORITHM_HPP_
#define HPS_ALGORITHM_HPP_

#include <p4est.h>
#include "Quadtree.hpp"
#include "EllipticForestApp.hpp"
#include "DataCache.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"

namespace EllipticForest {

template<typename PatchType, typename PatchGridType, typename PatchSolverType, typename NumericalType>
class HPSAlgorithm {

public:

    Quadtree<PatchType>* quadtree;
    p4est_t* p4est;
    PatchType& rootPatch;
    DataCache<Vector<NumericalType>> vectorCache;
    DataCache<Matrix<NumericalType>> matrixCache;

    HPSAlgorithm(PatchType& rootPatch) :
        rootPatch(rootPatch),
        p4est(nullptr)
            {}

    virtual void setupStage(p4est_t* p4est) {

        // Set internal p4est
        p4est = p4est;
        
        // Get app and log
        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Setup Stage");

        app.addTimer("setup-stage");
        app.timers["setup-stage"].start();

        // Build quadtree from p4est_t tree
        quadtree = new Quadtree(p4est);
        quadtree->build(rootPatch, [&](T& parentPatch, std::size_t level, std::size_t index){
            // Function to init patch from parent patch
            // Needs to be part of Patch derived class
        });

        // Set p4est ID (leaf level IDs) on patches
        int currentID = 0;
        quadtree->traversePostOrder([&](PatchType& patch){
            // if (patch.isLeaf()) {
            //     patch.leafID = currentID++;
            // }
            // else {
            //     patch.leafID = -1;
            // }

            patch.isLeaf() ? patch.leafID = currentID : patch.leafID = -1;
        });

        // Set D2N matrices on leaf
        quadtree->traversePostOrder([&](PatchType& patch){
            if (patch.isLeaf()) {
                PatchSolverBase solver;
                if (std::get<bool>(app.options["cache-operators"])) {
                    if (!matrixCache.contains("T_leaf")) {
                        matrixCache["T_leaf"] = solver.buildD2N
                    }
                }
                patch.matrixT() = matrixCache["T_leaf"];
            }
            else {
                patch.matrixT() = solver.buildD2N(patch.grid);
            }
        })

        app.timers["setup-stage"].stop();
        app.log("End HPS Setup Stage");

    }

    virtual void buildStage() {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Build Stage");
        app.addTimer("build-stage");
        app.timers["build-stage"].start();

        quadtree->merge([&](PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega){
            merge4to1(tau, alpha, beta, gamma, omega);
        });

        app.timers["build-stage"].stop();
        app.log("End HPS Build Stage");

    }

    virtual void upwardsStage() {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Upwards Stage");
        app.addTimer("upwards-stage");
        app.timers["upwards-stage"].start();

        quadtree->traversePostOrder([&](PatchType& patch){
            setParticularData(patch);
        });

        quadtree->merge([&](PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega){
            upwards4to1(tau, alpha, beta, gamma, omega);
        });

        app.timers["upwards-stage"].stop();
        app.log("End HPS Upwards Stage");

    }

    virtual void solveStage(Vector<int> dirichletData) {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Solve Stage");
        app.addTimer("solve-stage");
        app.timers["solve-stage"].start();

        // Set Dirichlet data on root patch
        quadtree->root().vectorG() = dirichletData;

        quadtree->split([&](PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega){
            split1to4(tau, alpha, beta, gamma, omega);
        });

        quadtree->traversePreOrder([&](PatchType& patch){
            leafSolve(patch);
        });

        app.timers["solve-stage"].stop();
        app.log("End HPS Solve Stage");

    }

    virtual void merge4to1(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        // app.log("Merging:");
        // app.log("  alpha = %i", alpha.globalID);
        // app.log("  beta = %i", beta.globalID);
        // app.log("  gamma = %i", gamma.globalID);
        // app.log("  omega = %i", omega.globalID);
        // app.log("  tau = %i", tau.globalID);

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

    virtual void setParticularData(PatchType& patch) {

        if (patch.isLeaf()) {
            // Create solver
            PatchSolverType solver;

            // Set RHS data on leaf patches
            PatchBase& grid = patch.grid;
            patch.f = Vector<double>(grid.nPointsX() * grid.nPointsY());
            for (auto i = 0; i < grid.nPointsX(); i++) {
                double x = grid(XDIM, i);
                for (auto j = 0; j < grid.nPointsY(); j++) {
                    double y = grid(YDIM, j);
                    int index = j + i*grid.nPointsY();
                    patch.f[index] = pde_.f(x, y);
                }
            }

            // Set Neumann data for particular solution
            Vector<double> gZero(2*grid.nPointsX() + 2*grid.nPointsY(), 0);
            patch.h = solver.mapD2N(grid, gZero, patch.f);
        }

    }

    virtual void upwards4to1(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            // app.log("Upwards:");
            // app.log("  alpha = %i", alpha.globalID);
            // app.log("  beta = %i", beta.globalID);
            // app.log("  gamma = %i", gamma.globalID);
            // app.log("  omega = %i", omega.globalID);
            // app.log("  tau = %i", tau.globalID);

            // Steps for the upwards stage (private member functions)
            coarsenUpwards_(tau, alpha, beta, gamma, omega);
            createIndexSets_(tau, alpha, beta, gamma, omega);
            createMatrixBlocks_(tau, alpha, beta, gamma, omega);
            mergeX_(tau, alpha, beta, gamma, omega);
            mergeW_(tau, alpha, beta, gamma, omega);
            mergeH_(tau, alpha, beta, gamma, omega);
            reorderOperatorsUpwards_(tau, alpha, beta, gamma, omega);
        }

        return;
    }

    virtual void split1to4(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        // app.log("Splitting:");
        // app.log("  tau = %i", tau.globalID);
        // app.log("  alpha = %i", alpha.globalID);
        // app.log("  beta = %i", beta.globalID);
        // app.log("  gamma = %i", gamma.globalID);
        // app.log("  omega = %i", omega.globalID);

        // Steps for the split (private member functions)
        uncoarsen_(tau, alpha, beta, gamma, omega);
        applyS_(tau, alpha, beta, gamma, omega);

        return;
    }

    virtual void leafSolve(PatchType& patch) {

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

    // Steps for the merge
    Vector<int> tagPatchesForCoarsening_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {
        
        std::vector<PatchGridType*> grids = {&alpha.grid, &beta.grid, &gamma.grid, &omega.grid};
        std::vector<int> sides(4);
        std::vector<int> tags(4);

        for (auto i = 0; i < 4; i++) { sides[i] = grids[i]->nPointsX(); }   // Get vector of side lengths
        int minSide = *std::min_element(sides.begin(), sides.end());        // Get minimum side length
        for (auto i = 0; i < 4; i++) { tags[i] = (sides[i] / minSide) - 1; }      // Get tags based on side lengths

        return {tags};
    }

    void coarsen_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patchPointers = {&alpha, &beta, &gamma, &omega};
        Vector<int> tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);
        int maxTag = *std::max_element(tags.data().begin(), tags.data().end());
        while (maxTag > 0) {
            for (auto i = 0; i < 4; i++) {
                if (tags[i] > 0) {
                    PatchGridType fineGrid = patchPointers[i]->grid;
                    PatchGridType coarseGrid(fineGrid.nPointsX()/2, fineGrid.nPointsY()/2, fineGrid.xLower(), fineGrid.xUpper(), fineGrid.yLower(), fineGrid.yUpper());
                    // int nFine = fineGrid.nPointsX();
                    // int nCoarse = coarseGrid.nPointsX();
            
                    // InterpolationMatrixFine2Coarse<double> L21Side(nCoarse);
                    // std::vector<Matrix<double>> L21Diagonals = {L21Side, L21Side, L21Side, L21Side};
                    // Matrix<double> L21Patch = blockDiagonalMatrix(L21Diagonals);

                    // InterpolationMatrixCoarse2Fine<double> L12Side(nFine);
                    // std::vector<Matrix<double>> L12Diagonals = {L12Side, L12Side, L12Side, L12Side};
                    // Matrix<double> L12Patch = blockDiagonalMatrix(L12Diagonals);

                    // patchPointers[i]->T = L21Patch * patchPointers[i]->T;
                    // patchPointers[i]->T = patchPointers[i]->T * L12Patch;
                    PatchSolverType solver;
                    patchPointers[i]->T = solver.buildD2N(coarseGrid);

                    patchPointers[i]->grid = coarseGrid;
                    patchPointers[i]->nCoarsens++;
                }
            }
            tags = tagPatchesForCoarsening_(tau, alpha, beta, gamma, omega);
            maxTag = *std::max_element(tags.data().begin(), tags.data().end());
        }

        return;
    }

    void createIndexSets_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check that all children patches are the same size (should be handled from the coarsening step if not)
        int nAlpha = alpha.grid.nPointsX();
        int nBeta = beta.grid.nPointsX();
        int nGamma = gamma.grid.nPointsX();
        int nOmega = omega.grid.nPointsX();
        Vector<int> n = {nAlpha, nBeta, nGamma, nOmega};
        if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
            throw std::invalid_argument("[EllipticForest::FISHPACK::FISHPACKHPSMethod::createIndexSets_] Size of children patches are not the same; something probably went wrong with the coarsening...");
        }

        int nSide = alpha.grid.nPointsX();

        Vector<int> I_W = vectorRange(0, nSide-1);
        Vector<int> I_E = vectorRange(nSide, 2*nSide - 1);
        Vector<int> I_S = vectorRange(2*nSide, 3*nSide - 1);
        Vector<int> I_N = vectorRange(3*nSide, 4*nSide - 1);

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

    void mergeX_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create diagonals
        Matrix<double> T_diag1 = T_ag_ag - T_ga_ga;
        Matrix<double> T_diag2 = T_bo_bo - T_ob_ob;
        Matrix<double> T_diag3 = T_ab_ab - T_ba_ba;
        Matrix<double> T_diag4 = T_go_go - T_og_og;
        std::vector<Matrix<double>> diag = {T_diag1, T_diag2, T_diag3, T_diag4};

        // Create row and column block index starts
        std::vector<std::size_t> rowStarts = { 0, T_diag1.nRows(), T_diag1.nRows() + T_diag2.nRows(), T_diag1.nRows() + T_diag2.nRows() + T_diag3.nRows() };
        std::vector<std::size_t> colStarts = { 0, T_diag1.nCols(), T_diag1.nCols() + T_diag2.nCols(), T_diag1.nCols() + T_diag2.nCols() + T_diag3.nCols() };
        
        // Create matrix and set blocks
        tau.matrixX() = blockDiagonalMatrix(diag);
        tau.matrixX().setBlock(rowStarts[0], colStarts[2], T_ag_ab);
        tau.matrixX().setBlock(rowStarts[0], colStarts[3], T_ga_go);
        tau.matrixX().setBlock(rowStarts[1], colStarts[2], T_bo_ba);
        tau.matrixX().setBlock(rowStarts[1], colStarts[3], T_ob_og);
        tau.matrixX().setBlock(rowStarts[2], colStarts[0], T_ab_ag);
        tau.matrixX().setBlock(rowStarts[2], colStarts[1], T_ba_bo);
        tau.matrixX().setBlock(rowStarts[3], colStarts[0], T_go_ga);
        tau.matrixX().setBlock(rowStarts[3], colStarts[1], T_og_ob);

        return;
    }

    void mergeS_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create right hand side
        std::size_t nRows = T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() + T_go_gt.nRows();
        std::size_t nCols = T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() + T_ob_ot.nCols();
        Matrix<double> S_RHS(nRows, nCols, 0);

        std::vector<std::size_t> rowStarts = { 0, T_ag_at.nRows(), T_ag_at.nRows() + T_bo_bt.nRows(), T_ag_at.nRows() + T_bo_bt.nRows() + T_ab_at.nRows() };
        std::vector<std::size_t> colStarts = { 0, T_ag_at.nCols(), T_ag_at.nCols() + T_bo_bt.nCols(), T_ag_at.nCols() + T_bo_bt.nCols() + T_ga_gt.nCols() };

        S_RHS.setBlock(rowStarts[0], colStarts[0], T_ag_at);
        S_RHS.setBlock(rowStarts[0], colStarts[2], T_ga_gt);
        S_RHS.setBlock(rowStarts[1], colStarts[1], T_bo_bt);
        S_RHS.setBlock(rowStarts[1], colStarts[3], T_ob_ot);
        S_RHS.setBlock(rowStarts[2], colStarts[0], T_ab_at);
        S_RHS.setBlock(rowStarts[2], colStarts[1], T_ba_bt);
        S_RHS.setBlock(rowStarts[3], colStarts[2], T_go_gt);
        S_RHS.setBlock(rowStarts[3], colStarts[3], T_og_ot);
        
        // Solve to set S_tau
        tau.matrixS() = solve(tau.X, S_RHS);

        return;
    }

    void mergeT_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Create left hand side
        std::vector<Matrix<double>> diag = {T_at_at, T_bt_bt, T_gt_gt, T_ot_ot};
        Matrix<double> T_LHS = blockDiagonalMatrix(diag);

        // Create right hand side
        std::size_t nRows = T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() + T_ot_ob.nRows();
        std::size_t nCols = T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() + T_gt_go.nCols();
        tau.matrixH() = Matrix<double>(nRows, nCols, 0);

        std::vector<std::size_t> rowStarts = { 0, T_at_ag.nRows(), T_at_ag.nRows() + T_bt_bo.nRows(), T_at_ag.nRows() + T_bt_bo.nRows() + T_gt_ga.nRows() };
        std::vector<std::size_t> colStarts = { 0, T_at_ag.nCols(), T_at_ag.nCols() + T_bt_bo.nCols(), T_at_ag.nCols() + T_bt_bo.nCols() + T_at_ab.nCols() };

        tau.matrixH().setBlock(rowStarts[0], colStarts[0], T_at_ag);
        tau.matrixH().setBlock(rowStarts[0], colStarts[2], T_at_ab);
        tau.matrixH().setBlock(rowStarts[1], colStarts[1], T_bt_bo);
        tau.matrixH().setBlock(rowStarts[1], colStarts[2], T_bt_ba);
        tau.matrixH().setBlock(rowStarts[2], colStarts[0], T_gt_ga);
        tau.matrixH().setBlock(rowStarts[2], colStarts[3], T_gt_go);
        tau.matrixH().setBlock(rowStarts[3], colStarts[1], T_ot_ob);
        tau.matrixH().setBlock(rowStarts[3], colStarts[3], T_ot_og);

        // Compute and set T_tau
        Matrix<double> T_RHS = tau.matrixH() * tau.matrixS();
        tau.matrixT() = T_LHS + T_RHS;

        return;
    }

    void reorderOperators_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Form permutation vector and block sizes
        int nSide = alpha.grid.nPointsX();
        Vector<int> pi_noChange = {0, 1, 2, 3};
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes1(4, nSide);
        Vector<int> blockSizes2(8, nSide);

        // Permute S and T
        tau.matrixS() = tau.matrixS().blockPermute(pi_noChange, pi_WESN, blockSizes1, blockSizes2);
        tau.matrixT() = tau.matrixT().blockPermute(pi_WESN, pi_WESN, blockSizes2, blockSizes2);

        return;
    }

    void mergePatch_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // TODO!

    }


    // Steps for the upwards stage
    void coarsenUpwards_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Check for adaptivity
        std::vector<PatchType*> patchPointers = {&alpha, &beta, &gamma, &omega};

        for (auto i = 0; i < 4; i++) {
            int nCoarsens = patchPointers[i]->nCoarsens;
            for (auto n = 0; n < nCoarsens; n++) {
                PatchGridType coarseGrid = patchPointers[i]->grid;
                PatchGridType fineGrid(coarseGrid.nPointsX()*2, coarseGrid.nPointsY()*2, coarseGrid.xLower(), coarseGrid.xUpper(), coarseGrid.yLower(), coarseGrid.yUpper());
                int nFine = fineGrid.nPointsX() * pow(2,nCoarsens-n-1);
                int nCoarse = coarseGrid.nPointsX() * pow(2,nCoarsens-n-1);
            
                InterpolationMatrixFine2Coarse<double> L21Side(nCoarse);
                std::vector<Matrix<double>> L21Diagonals = {L21Side, L21Side, L21Side, L21Side};
                Matrix<double> L21Patch = blockDiagonalMatrix(L21Diagonals);
                patchPointers[i]->vectorH() = L21Patch * patchPointers[i]->vectorH();
            }
        }

        return;
    }

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

        Vector<double> hDiff_gamma_alpha = h_ga - h_ag;
        Vector<double> hDiff_omega_beta = h_ob - h_bo;
        Vector<double> hDiff_beta_alpha = h_ba - h_ab;
        Vector<double> hDiff_omega_gamma = h_og - h_go;

        Vector<double> hDiff = concatenate({
            hDiff_gamma_alpha,
            hDiff_omega_beta,
            hDiff_beta_alpha,
            hDiff_omega_gamma
        });

        // Compute and set w_tau
        tau.vectorW() = solve(tau.matrixX(), hDiff);

        return;
    }

    void mergeH_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Compute and set h_tau
        tau.vectorH() = tau.matrixH() * tau.vectorW();

        // Update with boundary h
        Vector<double> h_alpha_tau = alpha.vectorH()(IS_alpha_tau_);
        Vector<double> h_beta_tau = beta.vectorH()(IS_beta_tau_);
        Vector<double> h_gamma_tau = gamma.vectorH()(IS_gamma_tau_);
        Vector<double> h_omega_tau = omega.vectorH()(IS_omega_tau_);
        Vector<double> hUpdate = concatenate({
            h_alpha_tau,
            h_beta_tau,
            h_gamma_tau,
            h_omega_tau
        });
        tau.vectorH() += hUpdate;

        return;
    }

    void reorderOperatorsUpwards_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        int nAlpha = alpha.grid.nPointsX();
        int nBeta = beta.grid.nPointsX();
        int nGamma = gamma.grid.nPointsX();
        int nOmega = omega.grid.nPointsX();
        Vector<int> n = {nAlpha, nBeta, nGamma, nOmega};
        if (!std::equal(n.data().begin()+1, n.data().end(), n.data().begin())) {
            throw std::invalid_argument("[EllipticForest::FISHPACK::FISHPACKHPSMethod::createIndexSets_] Size of children patches are not the same; something probably went wrong with the coarsening...");
        }

        // Form permutation vector and block sizes
        int nSide = alpha.grid.nPointsX();
        Vector<int> pi_WESN = {0, 4, 2, 6, 1, 3, 5, 7};
        Vector<int> blockSizes(8, nSide);

        // Reorder
        tau.vectorH() = tau.vectorH().blockPermute(pi_WESN, blockSizes);
    }


    // Steps for the split
    void uncoarsen_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        EllipticForestApp& app = EllipticForestApp::getInstance();

        for (auto n = 0; n < tau.nCoarsens; n++) {
            PatchGridType coarseGrid = tau.grid;
            PatchGridType fineGrid(coarseGrid.nPointsX()*2, coarseGrid.nPointsY()*2, coarseGrid.xLower(), coarseGrid.xUpper(), coarseGrid.yLower(), coarseGrid.yUpper());

            int nFine = fineGrid.nPointsX();
            int nCoarse = coarseGrid.nPointsX();

            InterpolationMatrixCoarse2Fine<double> L12Side(nFine);
            std::vector<Matrix<double>> L12Diagonals = {L12Side, L12Side, L12Side, L12Side};
            Matrix<double> L12Patch = blockDiagonalMatrix(L12Diagonals);
            tau.vectorG() = L12Patch * tau.vectorG();

            tau.grid = fineGrid;
        }

        return;
    }

    void applyS_(PatchType& tau, PatchType& alpha, PatchType& beta, PatchType& gamma, PatchType& omega) {

        // Apply solution operator to get interior of tau
        Vector<double> u_tau_interior = tau.matrixS() * tau.vectorG();

        // Apply non-homogeneous contribution
        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            u_tau_interior = u_tau_interior + tau.vectorW();
        }

        // Extract components of interior of tau
        int nSide = alpha.grid.nPointsX();
        Vector<double> g_alpha_gamma = u_tau_interior.getSegment(0*nSide, nSide);
        Vector<double> g_beta_omega = u_tau_interior.getSegment(1*nSide, nSide);
        Vector<double> g_alpha_beta = u_tau_interior.getSegment(2*nSide, nSide);
        Vector<double> g_gamma_omega = u_tau_interior.getSegment(3*nSide, nSide);

        // Extract components of exterior of tau
        Vector<double> g_alpha_W = tau.vectorG().getSegment(0*nSide, nSide);
        Vector<double> g_gamma_W = tau.vectorG().getSegment(1*nSide, nSide);
        Vector<double> g_beta_E = tau.vectorG().getSegment(2*nSide, nSide);
        Vector<double> g_omega_E = tau.vectorG().getSegment(3*nSide, nSide);
        Vector<double> g_alpha_S = tau.vectorG().getSegment(4*nSide, nSide);
        Vector<double> g_beta_S = tau.vectorG().getSegment(5*nSide, nSide);
        Vector<double> g_gamma_N = tau.vectorG().getSegment(6*nSide, nSide);
        Vector<double> g_omega_N = tau.vectorG().getSegment(7*nSide, nSide);

        // Set child patch Dirichlet data
        alpha.vectorG() = concatenate({g_alpha_W, g_alpha_beta, g_alpha_S, g_alpha_gamma});
        beta.vectorG() = concatenate({g_alpha_beta, g_beta_E, g_beta_S, g_beta_omega});
        gamma.vectorG() = concatenate({g_gamma_W, g_gamma_omega, g_alpha_gamma, g_gamma_N});
        omega.vectorG() = concatenate({g_gamma_omega, g_omega_E, g_beta_omega, g_omega_N});

        return;
    }


};

template<typename PatchType, typename NumericalType>
class HomogeneousHPSMethod : public HPSAlgorithmBase<PatchType, NumericalType> {

public:

    virtual void run() override {

        this->preSetupHook();
        this->setupStage();
        this->postSetupHook();

        this->preBuildHook();
        this->buildStage();
        this->postBuildHook();

        this->preSolveHook();
        this->solveStage();
        this->postSolveHook();

    }

protected:

    // virtual void preSetupHook() {}
    virtual void setupStage() override = 0;
    // virtual void postSetupHook() {}

    // virtual void preBuildHook() {}
    // virtual void buildStage();
    // virtual void postBuildHook() {}

    // virtual void preSolveHook() {}
    // virtual void solveStage();
    // virtual void postSolveHook() {}

    // virtual void postProcess() {}

};

} // NAMESPACE : EllipticForest

#endif // HPS_ALGORITHM_HPP_