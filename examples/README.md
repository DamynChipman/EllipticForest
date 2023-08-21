# EllipticForest Examples

## Hello

A "Hello World" test that imports the `EllipticForestApp` header, creates the app, checks for `matplotlibcpp`, then ends.

## Patch Solver

Solves Poisson's equation on a single patch. Solves the BVP with the PETSc patch solver.

## Elliptic Single

Solves an elliptic PDE with the adaptive HPS method.

## Elliptic Multiple

Solves an elliptic PDE with a sweep of refinement levels and patch sizes to check convergence of EllipticForest.

## Thermal

Solves a variable coefficient heat equation with EllipticForest to show the ability to split the build and solve stages.