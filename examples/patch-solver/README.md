# EllipticForest Examples

## Patch-Solver

Runs a convergence sweep of the PETSc patch solver with a second order patch solver.

## Usage

```Bash
./patch-solver
```

## Output

```bash
[EllipticForest] Welcome to EllipticForest!
[EllipticForest 0] N =    8, Error = 6.62291775e-03
[EllipticForest 0] N =   16, Error = 1.65048768e-03
[EllipticForest 0] N =   32, Error = 4.12493989e-04
[EllipticForest 0] N =   64, Error = 1.03084267e-04
[EllipticForest 0] N =  128, Error = 2.57714346e-05
[EllipticForest 0] N =  256, Error = 6.44288352e-06
[EllipticForest 0] N =  512, Error = 1.61072306e-06
[EllipticForest 0] N = 1024, Error = 4.02680830e-07
[EllipticForest 0] Convergence Order = 2.0000
[EllipticForest] End of app life cycle, finalizing...
[EllipticForest] Timers: 
[EllipticForest]   app-lifetime : 50.8172 [sec]
[EllipticForest] Done!
```