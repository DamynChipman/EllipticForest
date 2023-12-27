---
title: 'EllipticForest: A Direct Solver for Elliptic Partial Differential Equations on Adaptive Meshes'
tags:
  - C++
  - numerical linear algebra
  - partial differential equations
  - adaptive mesh refinement
authors:
  - name: Damyn Chipman
    orcid: 0000-0001-6600-3720
    affiliation: 1 # (Multiple affiliations must be quoted)
    equal-contrib: true
    corresponding: true
  - name: Donna Calhoun
    orcid: 0000-0002-6005-4575
    affiliation: 1
affiliations:
 - name: Boise State University, USA
   index: 1
date: 1 January 2024
bibliography: paper.bib
---

# Summary

EllipticForest is a software library with utilities to solve elliptic partial differential equations (PDEs) with adaptive mesh refinement (AMR) using a direct matrix factorization. It is a quadtree-adaptive implementation of the Hierarchical Poincaré-Steklov (HPS) method [@gillman2014direct]. The HPS method is a direct method for solving elliptic PDEs based on the recursive merging of Poincaré-Steklov operators [@quarteroni1991theory]. EllipticForest features coupling with the parallel and highly efficient mesh library `p4est` [@burstedde2011p4est] for mesh adaptivity and mesh management. Distributed memory parallelism is implemented through the Message Passing Interface (MPI) [@walker1996mpi], which is a first for the HPS method. The primary feature of EllipticForest is the modularity for users to extend the solver interface to custom solvers at the leaf-level. By default, EllipticForest implements fast cyclic-reduction methods as found in the FISHPACK [@swarztrauber1999fishpack] library and updated in the FISHPACK90 [@adams2016fishpack90] code. In addition, for more general elliptic problems, EllipticForest wraps solvers from the PDE solver library PETSc [@anl2023petsc].

Similar to other direct methods, the HPS method is comprised of two stages: a build stage and a solve stage. In the build stage, a set of solution operators are formed that act as the factorization of the system matrix corresponding to the discretization stencil. In the solve stage, the factorization is applied to a right-hand side vector corresponding to boundary and non-homogeneous data to solve the problem with linear complexity $\mathcal{O}(N)$ where $N$ is the size of the system matrix. The advantages of this approach over iterative methods such as conjugate gradient and multi-grid methods include the ability to apply the factorization to multiple right-hand sides.

# Statement of need

The primary novelty of EllipticForest as software is the implementation of the HPS method for user extension. The quadtree-adaptive implementation of the HPS method as featured in [TODO: Cite paper 1] and it's parallel counterpart in [TODO: Cite paper 2] is novel in the field of fast solvers for AMR. This paper highlights the software implementation including  the user-friend interface to the HPS method and the ability for users to extend the solver interface using object-oriented programming (OOP) paradigms. Currently, all other implementations of the HPS method are MATLAB codes that are tailored to each research groups' needs and research endeavors. EllipticForest is the first C++ software ready for user-extension and coupling with other scientific solver libraries.

# Methods Overview

## Elliptic Partial Differential Equations

The general form of elliptic PDEs that EllipticForest is tailored to solve is the following:

$$
\alpha(x,y) \nabla \cdot \Big[ \beta(x,y) \nabla u(x,y) \Big] + \lambda(x,y) u(x,y) = f(x,y)
$$

where $\alpha(x,y)$, $\beta(x,y)$, $\lambda(x,y)$, and $f(x,y)$ are provided functions in $x$ and $y$ and the goal is to solve for $u(x,y)$. Currently, EllipticForest solves the above problem in a rectangular domain $\Omega = [x_L, x_U] \times [y_L, y_U]$. The above PDE is discretized using a finite-volume approach using a standard five-point stencil yielding a second-order accurate solution. Extensions to higher order are possible and have been highlighted in other papers [TODO: Cite other papers].

## Adaptive Mesh Refinement

[TODO]

# Software Overview

In this section, we will outline the steps users need to take in order to use EllipticForest to solve a PDE as well as the options available for extension via inheritance.

## Usage

### Patches

The fundamental building blocks of EllipticForest are patches. A patch is a subset of the domain and holds information such as solution matrices, data vectors, and discrete grids. A leaf-patch is a patch that...

### Mesh

Solving an elliptic PDE with EllipticForest starts by creating a mesh object `Mesh`. `Mesh` is templated to the...

## Extension

### Inheritance

## Complete Example

Below is a snippet of code that highlights how to use EllipticForest to solve an elliptic PDE on an adaptive mesh.

```C++
#include <cmath>
#include <iostream>
#include <string>
#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>
using namespace EllipticForest;

double uFunction(double x, double y) {
    return sin(x) + sin(y);
}

double fFunction(double x, double y) {
    return -uFunction(x, y);
}

int main(int argc, char** argv) {
    // Initialize app and MPI
    EllipticForestApp app(&argc, &argv);
    MPI::MPIObject mpi(MPI_COMM_WORLD);

    // Setup options
    double threshold = 1.2;
    int n_solves = 1;
    int min_level = 0;
    int max_level = 7;
    double x_lower = -10.0;
    double x_upper = 10.0;
    double y_lower = -10.0;
    double y_upper = 10.0;
    int nx = 8;
    int ny = 8;

    // Create grid and patch prototypes
    FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // Create node factory and mesh
    FiniteVolumeNodeFactory node_factory(MPI_COMM_WORLD);
    Mesh<FiniteVolumePatch> mesh{};

    // Refine mesh based on right-hand side function
    mesh.refineByFunction(
        [&](double x, double y){
            double f = fFunction(x, y);
            return fabs(f) > threshold;
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        node_factory
    );

    // Create patch solver
    FiniteVolumeSolver solver{};
    solver.solver_type = FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = [&](double x, double y){
        return 1.0;
    };
    solver.beta_function = [&](double x, double y){
        return 1.0;
    };
    solver.lambda_function = [&](double x, double y){
        return 0.0;
    };

    // Create and run HPS solver
    // 1. Create the HPSAlgorithm instance
    HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double> HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // Begin solver loop; demonstrates ability to solve multiple times once build stage is done
    for (auto n = 0; n < n_solves; n++) {
        // 4. Call the upwards stage; provide a callback to set load data on leaf patches
        HPS.upwardsStage([&](double x, double y){
            return fFunction(x, y);
        });

        // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
        HPS.solveStage([&](int side, double x, double y, double* a, double* b){
            *a = 1.0;
            *b = 0.0;
            return uFunction(x, y);
        });
    }

    // Write solution and functions to file
    if (vtk_flag) {
        // Extract out solution and right-hand side data stored on leaves
        Vector<double> u_mesh{};
        u_mesh.name() = "u_soln";
        Vector<double> f_mesh{};
        f_mesh.name() = "f_rhs";
        
        // Traverse mesh to extract `u` and `f` from patches
        mesh.quadtree.traversePreOrder([&](Node<FiniteVolumePatch>* node){
            if (node->leaf) {
                auto& patch = node->data;
                auto& grid = patch.grid();

                u_mesh.append(patch.vectorU());
                f_mesh.append(patch.vectorF());
            }
            return 1;
        });

        // Write VTK files:
        //      "elliptic-mesh.pvtu"            : Parallel header file for mesh and data
        //      "elliptic-quadtree.pvtu"        : p4est quadtree structure
        mesh.toVTK("elliptic");

    }

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}
```

Running the above code will generate a parallel, unstructured VTK format file that can be viewed with standard scientific data visualization software such as ParaView [@paraview] or VisIt [@HPV:VisIt]. Figure \autoref{fig:poisson_solution} shows this done with VisIt.

![Solution of Poisson equation on a quadtree mesh using EllipticForest.\label{fig:poisson_solution}](examples/elliptic-single/output.png)

# References