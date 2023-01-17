# EllipticForest

A Quadtree-Adatpive implementation of the Hierarchical Poincaré-Steklov (HPS) method for solving elliptic partial differential equations.

## Features

- A flexible implementation for use with different patch solvers
- Ability to solve Laplace's and Poisson's equation (with variable coefficient elliptic forthcoming)

## Usage

See `examples` for usage.

## Installation

`EllipticForest` relies on several external packages for the linear algebra and patch solvers:

    - `p4est` : Library for managing a collection of quadtrees or octrees for adaptive meshes.
    - `petsc` : Scalable linear algebra package for solving PDEs on distributed meshes.
    - `FISHPACK` : A FORTRAN70 code for solving elliptic PDEs on strucutres meshes.
    - `matplotlibcpp` : A C++ wrapper for some `matplotlib` functionality.

Currently, `EllipticForest` requires the user to have most of these installed externally and then point the build script to the user's implementation. This will change in the next release with optional building.

An exmaple configure script looks like the following:

```
# --=== User Variables ===--
# HOME : Path to home directory
HOME=/Users/damynchipman

# PYTHON_ENV_PATH : Path to conda `HydroForest` environment directory
# NOTE: Also check ${HOME}/miniconda3 or ${HOME}/anaconda
PYTHON_ENV_PATH=${HOME}/miniforge3/envs/HydroForest

# PYTHON_VERSION : Version of Python in conda `EllipticForest` envrionment
PYTHON_VERSION=python3.9

# P4EST_PATH : Path to p4est install (i.e., ${P4EST_PATH}/include, ${P4EST_PATH}/lib, ...)
P4EST_PATH=${HOME}/packages/p4est/p4est_source_git/build/local

# PETSC_PATH : Path to PETSc install (i.e., ${PETSC_PATH}/include, ${PETSC_PATH}/lib, ...) 
PETSC_PATH=${HOME}/packages/petsc/petsc-build

# ELLIPTIC_FOREST : Absolute path to source code for HydroForest
ELLIPTIC_FOREST=${HOME}/packages/EllipticForest

# --=== Create Build Directory ===--
BUILD_DIR=build-$(git branch --show-current)
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# --=== CMake Configure ===--
cmake ${ELLIPTIC_FOREST} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_C_COMPILER=mpicc \
    -DPYTHON_ENV_PATH=${PYTHON_ENV_PATH} \
    -DPYTHON_VERSION=${PYTHON_VERSION} \
    -DPETSC_PATH=${PETSC_PATH} \
    -DP4EST_PATH=${P4EST_PATH}

echo "Now cd to $(pwd)/${BUILD_DIR} and run make to compile"
```