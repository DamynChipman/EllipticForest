# EllipticForest Examples

## Hello

A "Hello World" test that imports the `EllipticForestApp` header, creates the app, checks for `matplotlibcpp`, then ends.

```bash
cd hello
./hello
```

## Poisson

Solves a user defined Poisson problem (implemented by deriving from the base `EllipticForest::EllipticProblemBase` class). Runs a convergence analysis and generates error and timing plots.

```bash
cd poisson
./poisson
```

## Helmholtz

Solves a Helmholtz problem (implemented by deriving from the base `EllipticForest::EllipticProblemBase` class). Runs a parameter sweep of the patch size, the level of refinement, and varying values of `lambda`.

```bash
cd helmholtz
./helmholtz
```

## Toybox

Where I do a bunch of my testing and debugging. Right now, it has a refine and coarsen test for the adaptive quadtree features.

```bash
cd toybox
./toybox
```