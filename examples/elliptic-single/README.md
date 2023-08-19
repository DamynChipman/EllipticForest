# EllipticForest Examples

## Elliptic-Single

Solves Poisson's equation:

$$\Delta u = f$$

subject to Dirichlet boundary conditions provided by the exact solution.

By default, this is set to solve for the exact solution:

$$u(x,y) = sin(x) + sin(y)$$

thus,

$$f(x,y) = -sin(x) - sin(y) = -u(x,y).$$

EllipticForest solves this by creating a mesh and refining it according to the curvature of the
solution (i.e., the right-hand side function `f`). The build, upwards, and solve stages are used
to do the factorization and application of the solution operators. The solution is output to VTK
files to be viewed with your favorite visualization tool (VisIt is mine!)

## Usage

```Bash
mpirun -n <number_of_processes> ./elliptic-single
```

## Output

![](output.png)