#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <FISHPACK.hpp>
#include <P4est.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>

namespace ef = EllipticForest;

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

int main(int argc, char** argv) {

    // Create app
    ef::EllipticForestApp app(&argc, &argv);

    // Create four child grids and parent grid
    int N = atoi(argv[1]);
    int nx = N;
    int ny = N;
    double xLower = -1;
    double xMid = 0;
    double xUpper = 1;
    double yLower = -1;
    double yMid = 0;
    double yUpper = 1;
    ef::Petsc::PetscGrid gridAlpha(nx, ny, xLower, xMid, yLower, yMid);
    ef::Petsc::PetscGrid gridBeta(nx, ny, xMid, xUpper, yLower, yMid);
    ef::Petsc::PetscGrid gridGamma(nx, ny, xLower, xMid, yMid, yUpper);
    ef::Petsc::PetscGrid gridOmega(nx, ny, xMid, xUpper, yMid, yUpper);
    ef::Petsc::PetscGrid gridTau(2*nx, 2*ny, xLower, xUpper, yLower, yUpper);
    // ef::FISHPACK::FISHPACKFVGrid gridAlpha(nx, ny, xLower, xMid, yLower, yMid);
    // ef::FISHPACK::FISHPACKFVGrid gridBeta(nx, ny, xMid, xUpper, yLower, yMid);
    // ef::FISHPACK::FISHPACKFVGrid gridGamma(nx, ny, xLower, xMid, yMid, yUpper);
    // ef::FISHPACK::FISHPACKFVGrid gridOmega(nx, ny, xMid, xUpper, yMid, yUpper);
    ef::FISHPACK::FISHPACKFVGrid gridTauFISHPACK(2*nx, 2*ny, xLower, xUpper, yLower, yUpper);

    // Create four child patches and parent patch
    ef::Petsc::PetscPatch patchAlpha(gridAlpha);
    ef::Petsc::PetscPatch patchBeta(gridBeta);
    ef::Petsc::PetscPatch patchGamma(gridGamma);
    ef::Petsc::PetscPatch patchOmega(gridOmega);
    ef::Petsc::PetscPatch patchTau(gridTau);
    // ef::FISHPACK::FISHPACKPatch patchAlpha(gridAlpha);
    // ef::FISHPACK::FISHPACKPatch patchBeta(gridBeta);
    // ef::FISHPACK::FISHPACKPatch patchGamma(gridGamma);
    // ef::FISHPACK::FISHPACKPatch patchOmega(gridOmega);
    ef::FISHPACK::FISHPACKPatch patchTauFISHPACK(gridTauFISHPACK);

    // Create patch solver
    ef::Petsc::PetscPatchSolver petsc_solver;
    petsc_solver.setAlphaFunction([&](double x, double y){
        return 1.0;
    });
    petsc_solver.setBetaFunction([&](double x, double y){
        return 1.0;
    });
    petsc_solver.setLambdaFunction([&](double x, double y){
        return 0.0;
    });
    ef::FISHPACK::FISHPACKFVSolver fishpack_solver;

    // Create HPS instance
    ef::HPSAlgorithm<ef::Petsc::PetscGrid, ef::Petsc::PetscPatchSolver, ef::Petsc::PetscPatch, double> HPS;
    // ef::HPSAlgorithm<ef::FISHPACK::FISHPACKFVGrid, ef::FISHPACK::FISHPACKFVSolver, ef::FISHPACK::FISHPACKPatch, double> HPS;

    // Compute leaf level T
    patchAlpha.matrixT() = petsc_solver.buildD2N(gridAlpha);
    patchBeta.matrixT() = petsc_solver.buildD2N(gridBeta);
    patchGamma.matrixT() = petsc_solver.buildD2N(gridGamma);
    patchOmega.matrixT() = petsc_solver.buildD2N(gridOmega);

    // Compute parent DtN matrix from original patch
    ef::Matrix<double> T_patch_petsc = petsc_solver.buildD2N(patchTau.grid());
    ef::Matrix<double> T_patch_fishpack = fishpack_solver.buildD2N(patchTauFISHPACK.grid());

    // Merge 4-to-1
    HPS.merge4to1(patchTau, patchAlpha, patchBeta, patchGamma, patchOmega);
    ef::Matrix<double>& T_merge = patchTau.matrixT();

    // Compute error
    // ef::Matrix<double> T_diff_petsc = T_patch_petsc - T_merge;
    // ef::Matrix<double> T_diff_fishpack = T_patch_fishpack - T_merge;
    app.log("PETSc merged vs. PETSc parent error    = %16.6e", ef::matrixInfNorm(T_patch_petsc, T_merge));
    app.log("PETSc merged vs. FISHPACK parent error = %16.6e", ef::matrixInfNorm(T_patch_fishpack, T_merge));

    return EXIT_SUCCESS;
}