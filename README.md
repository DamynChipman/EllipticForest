# EllipticForest

A Quadtree-Adaptive implementation of the Hierarchical Poincaré-Steklov (HPS) method for solving elliptic partial differential equations.

## Features

- Solve elliptic PDEs on an adaptive mesh
- User-friendly implementation of the HPS method with VTK output
- An object-oriented code allows for user-expandability for additional patch solvers
- Ability to solve elliptic PDEs of the forms:
  - Laplace equation : $\Delta u(x,y) = 0$
  - Poisson equation : $\Delta u(x,y) = f(x,y)$
  - Helmholtz equation : $\Delta u(x,y) + \lambda u(x,y) = f(x,y)$
  - Variable coefficient elliptic equation : $\alpha(x,y) \nabla \cdot  \left[\beta(x,y) \nabla u(x,y)\right] + \lambda(x,y) u(x,y) = f(x,y)$

## Usage

Here is a minimum working example of using the HPS stages to solve an elliptic problem:

```C++
#include <EllipticForest.hpp>
#include <Patches/FiniteVolume/FiniteVolume.hpp>

int main(int argc, char** argv) {

    // ====================================================
    // Initialize app and MPI
    // ====================================================
    EllipticForest::EllipticForestApp app(&argc, &argv);
    EllipticForest::MPI::MPIObject mpi(MPI_COMM_WORLD);

    // ====================================================
    // Setup options
    // ====================================================
    bool cache_operators = false;
    bool homogeneous_rhs = false;
    bool vtk_flag = true;
    double threshold = 1.2;
    int n_solves = 1;
    int min_level = 0;
    int max_level = 7;
    double x_lower = -10.0;
    double x_upper = 10.0;
    double y_lower = -10.0;
    double y_upper = 10.0;
    int nx = 16;
    int ny = 16;

    // ====================================================
    // Create grid and patch prototypes
    // ====================================================
    EllipticForest::FiniteVolumeGrid grid(MPI_COMM_WORLD, nx, x_lower, x_upper, ny, y_lower, y_upper);
    EllipticForest::FiniteVolumePatch root_patch(MPI_COMM_WORLD, grid);

    // ====================================================
    // Create node factory and mesh
    // ====================================================
    EllipticForest::FiniteVolumeNodeFactory node_factory(MPI_COMM_WORLD);
    EllipticForest::Mesh<EllipticForest::FiniteVolumePatch> mesh{};
    mesh.refineByFunction(
        [&](double x, double y){
            double f = -(sin(x) + sin(y));
            return fabs(f) > threshold;
        },
        threshold,
        min_level,
        max_level,
        root_patch,
        node_factory
    );

    // ====================================================
    // Create patch solver
    // ====================================================
    EllipticForest::FiniteVolumeSolver solver{};
    solver.solver_type = EllipticForest::FiniteVolumeSolverType::FISHPACK90;
    solver.alpha_function = alphaFunction;
    solver.beta_function = betaFunction;
    solver.lambda_function = lambdaFunction;

    // ====================================================
    // Create and run HPS solver
    // ====================================================
    // 1. Create the HPSAlgorithm instance
    EllipticForest::HPSAlgorithm<EllipticForest::FiniteVolumeGrid, EllipticForest::FiniteVolumeSolver, EllipticForest::FiniteVolumePatch, double> HPS(MPI_COMM_WORLD, mesh, solver);

    // 2. Call the setup stage
    HPS.setupStage();

    // 3. Call the build stage
    HPS.buildStage();

    // 4. Call the upwards stage; provide a callback to set load data on leaf patches
    HPS.upwardsStage([&](double x, double y){
        return -(sin(x) + sin(y));
    });

    // 5. Call the solve stage; provide a callback to set physical boundary Dirichlet data on root patch
    HPS.solveStage([&](int side, double x, double y, double* a, double* b){
        *a = 1.0;
        *b = 0.0;
        return sin(x) + sin(y);
    });

    // All clean up is done in destructors
    return EXIT_SUCCESS;
}
```

Running the above (which is a selection from `examples/elliptic-single`) outputs Unstructured Mesh VTK files which can be visualized with ParaView, VisIt, etc.:

![](examples/elliptic-single/output.png)

## Configuring, Building, and Installing

`EllipticForest` relies on several external packages for the mesh adaptivity, linear algebra, and parallelism:

Required:
- `MPI` : Message-passing interface for distributed memory parallelism
- `BLAS` and `LAPACK` : Basic Linear Algebra Subprograms and Linear Algebra PACKage
- `FISHPACK90` [(GitHub)](https://github.com/DamynChipman/fishpack90) : A FORTRAN90 code for solving elliptic PDEs on structured grids (leaf patches).
- `p4est` [(GitHub)](https://github.com/cburstedde/p4est) : Library for managing a collection of quadtrees or octrees for adaptive meshes.
- `petsc` [(GitLab)](https://gitlab.com/petsc/petsc) : Scalable linear algebra package for solving PDEs on distributed meshes.

Optional:
- `matplotlibcpp` [(GitHub)](https://github.com/DamynChipman/matplotlib-cpp) : A C++ wrapper for some `matplotlib` functionality.
  
The only packages the user is required to have installed are `MPI`, `BLAS`, and `LAPACK`; `EllipticForest` will attempt to build `FISHPACK90`, `p4est`, and `petsc` internally if not provided externally. Because `petsc` is not built with CMake, it is built with CMake's `ExternalProject` which adds a lot of extra compilation and additional steps. Ideally, the user will install `petsc` themselves and provide the path to it as detailed below. Instructions for installing `petsc` can be found [here](https://petsc.org/release/install/) and can be done with `apt install`, Homebrew (Mac), and Spack.

The user may also provide paths to already installed versions of each of these packages and `EllipticForest` will use them accordingly.

`EllipticForest` uses CMake as the build system.

### Configuration Examples

#### Minimum Configuration

```bash
>>> cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DMPI_PATH=${PATH_TO_MPI}
```

#### Pre-Installed Packages

```bash
>>> cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DMPI_PATH=${PATH_TO_MPI} -DFISHPACK90_PATH=${PATH_TO_FISHPACK90} -DP4EST_PATH=${PATH_TO_P4EST} -DPETSC_PATH=${PATH_TO_PETSC}
```

#### `matplotlibcpp` Plotting Features

```bash
>>> cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DMPI_PATH=${PATH_TO_MPI} -DWITH_MATPLOTLIBCPP=true -DPYTHON_ENV_PATH=${PYTHON_ENV_PATH} -DPYTHON_VERSION=${PYTHON_VERSION}
```

### Building Examples

#### Minimum Building

```bash
>>> make
```

#### Testing Interface

`EllipticForest` uses GoogleTest and CTest for unit testing.

```bash
>>> make test
```

### Installation Examples

```bash
>>> make install
```

## Examples

There are some additional examples found in the `examples` directory. See the `README` therein for an overview of examples.

## References

[1] S. Balay, S. Abhyankar, M. F. Adams, S. Benson, J. Brown, P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener, V. Eijkhout, J. Faibussowitsch, W. D. Gropp, V. Hapla, T. Isaac, P. Jolivet, D. Karpeev, D. Kaushik, M. G. Knepley, F. Kong, S. Kruger, D. A. May, L. C. McInnes, R. T. Mills, L. Mitchell, T. Munson, J. E. Roman, K. Rupp, P. Sanan, J. Sarich, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, and J. Zhang. PETSc/TAO users manual. Technical Report ANL-21/39 - Revision 3.20, Argonne National Laboratory, 2023.

[2] C. Burstedde, L. C. Wilcox, and O. Ghattas. p4est: Scalable algorithms for parallel adaptive mesh refinement on forests of octrees. SIAM Journal on Scientific Computing, 33(3):1103–1133, 2011.

[3] A. Gillman and P.-G. Martinsson. A direct solver with O(N) complexity for variable coefficient ellip- tic PDEs discretized via a high-order composite spectral collocation method. SIAM J. Sci. Comput., 36(4):A2023–A2046, 2014.

[4] P. Martinsson. The hierarchical Poincar ́e-Steklov (HPS) solver for elliptic PDEs: A tutorial. arXiv preprint arXiv:1506.01308, 2015.

[5] P.-G. Martinsson. Fast direct solvers for elliptic PDEs. SIAM, 2019.
