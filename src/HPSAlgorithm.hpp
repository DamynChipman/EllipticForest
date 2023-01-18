#ifndef HPS_ALGORITHM_HPP_
#define HPS_ALGORITHM_HPP_

#include <p4est.h>
#include "Quadtree.hpp"
#include "EllipticForestApp.hpp"
#include "DataCache.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"

namespace EllipticForest {

template<typename QuadtreeNodeType, typename NumericalType>
class HPSAlgorithmBase {

public:

    virtual void run() {

        this->preSetupHook();
        this->setupStage();
        this->postSetupHook();

        this->preBuildHook();
        this->buildStage();
        this->postBuildHook();

        EllipticForestApp& app = EllipticForestApp::getInstance();
        if (!std::get<bool>(app.options["homogeneous-rhs"])) {
            this->preUpwardsHook();
            this->upwardsStage();
            this->postUpwardsHook();
        }

        this->preSolveHook();
        this->solveStage();
        this->postSolveHook();

    }

    Quadtree<QuadtreeNodeType>* quadtree;
    DataCache<Vector<NumericalType>> vectorCache;
    DataCache<Matrix<NumericalType>> matrixCache;

    virtual void preSetupHook() {}

    virtual void setupStage() = 0;

    virtual void postSetupHook() {}

    virtual void merge4to1(QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega) = 0;

    virtual void preBuildHook() {}

    virtual void buildStage() {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Build Stage");
        app.addTimer("build-stage");
        app.timers["build-stage"].start();

        quadtree->merge([&](QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega){
            merge4to1(tau, alpha, beta, gamma, omega);
        });

        app.timers["build-stage"].stop();
        app.log("End HPS Build Stage");

    }

    virtual void postBuildHook() {}

    virtual void setParticularData(QuadtreeNodeType& patch) = 0;

    virtual void upwards4to1(QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega) = 0;

    virtual void preUpwardsHook() {}

    virtual void upwardsStage() {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Upwards Stage");
        app.addTimer("upwards-stage");
        app.timers["upwards-stage"].start();

        quadtree->traversePostOrder([&](QuadtreeNodeType& patch){
            setParticularData(patch);
        });

        quadtree->merge([&](QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega){
            upwards4to1(tau, alpha, beta, gamma, omega);
        });

        app.timers["upwards-stage"].stop();
        app.log("End HPS Upwards Stage");

    }

    virtual void postUpwardsHook() {}

    virtual void split1to4(QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega) = 0;

    virtual void leafSolve(QuadtreeNodeType& patch) = 0;

    virtual void preSolveHook() {}

    virtual void solveStage() {

        EllipticForestApp& app = EllipticForestApp::getInstance();
        app.log("Begin HPS Solve Stage");
        app.addTimer("solve-stage");
        app.timers["solve-stage"].start();

        quadtree->split([&](QuadtreeNodeType& tau, QuadtreeNodeType& alpha, QuadtreeNodeType& beta, QuadtreeNodeType& gamma, QuadtreeNodeType& omega){
            split1to4(tau, alpha, beta, gamma, omega);
        });

        quadtree->traversePreOrder([&](QuadtreeNodeType& patch){
            leafSolve(patch);
        });

        app.timers["solve-stage"].stop();
        app.log("End HPS Solve Stage");

    }

    virtual void postSolveHook() {}

    virtual void postProcess() {}

};

template<typename QuadtreeNodeType, typename NumericalType>
class HomogeneousHPSMethod : public HPSAlgorithmBase<QuadtreeNodeType, NumericalType> {

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