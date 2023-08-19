# EllipticForest Examples

## Elliptic-Single

Solves a user-defined elliptic PDE in a rectangular domain on an adaptive mesh.

The PDE to solve is the following:

$\alpha(x,y) \nabla \cdot \big[ \beta(x,y) \nabla u(x,y) \big] + \lambda(x,y) u(x, y) = f(x,y), x,y \in \Omega=[x_{lower}, x_{upper}] \times [y_{lower}, y_{upper}]$

subject to Dirichlet boundary conditions:

$u(x,y) = g(x,y), x,y \in \Gamma = \partial \Omega$

or Neumann boundary conditions:

$\frac{\partial u(x,y)}{\partial n} = v(x,y), x,y \in \Gamma = \partial \Omega$

## Usage

```Bash
mpirun -n <number_of_processes> ./elliptic-single
```