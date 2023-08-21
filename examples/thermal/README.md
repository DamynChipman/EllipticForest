# EllipticForest Examples

## Thermal

This example solves the variable coefficient heat equation:

$$\frac{\partial u}{\partial t} = \nabla \cdot (\beta \nabla u) + \frac{q_{volume}}{k}$$

where:

$u = u(x,y,t) =$ Temperature

$\beta = \beta(x,y) =$ Thermal diffusivity

$q_{volume} = q_{volume}(x,y,t) =$ Volumetric heat source/sink

$k =$ Thermal conductivity

subject to the following boundary conditions:

$$u = g(x,y,t) , x,y \in \Omega_{Dirichlet}$$
$$\frac{\partial u}{\partial n} = v(x,y,t), x,y \in Omega_{Neumann}$$

with zero initial conditions.
```
             du/dn = 0
         ________________
         |               |
         |               |
   u=T_L |               | u=T_R
         |               |
         |               |
         |_______________|
             du/dn = 0
```
This is solved via backward Euler to obtain the following implicit scheme:

$$\nabla \cdot ( \beta \nabla u^{n+1}) - \lambda u^{n+1} = -\lambda u^{n} - \frac{q_{volume}}{k}$$

where:

$$\lambda = \frac{1}{dt}$$

which is solved using EllipticForest. The mesh is built with according to a user defined
criteria (defaults to refinement at the edges of the domain). The set of solution operators
are built and then used for each time step, resulting in a very fast time per timestep.

The user can mess around with most pieces of this equation to your heart's desire! Boundary
conditions are imposed in the functions `uWest`, `uEast`, `dudnSouth`, and `dudnNorth`. The
volumetric source/sinks can be added in the function `sources`. The variable thermal diffusivity
can be changed in `betaFunction`.

## Usage

```Bash
mpirun -n <number_of_processes> ./thermal
```