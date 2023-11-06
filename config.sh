#!/bin/sh

# HOME : Path to home directory
HOME=/Users/damynchipman

# --=== REQUIRED User Variables ===--
# ELLIPTIC_FOREST : Absolute path to source code for EllipticForest
ELLIPTIC_FOREST=${HOME}/packages/EllipticForest

# P4EST_PATH : Path to p4est install (i.e., ${P4EST_PATH}/include, ${P4EST_PATH}/lib, ...)
P4EST_PATH=${HOME}/packages/p4est/p4est/build/local

# MPI_PATH : Directory with MPI headers (i.e. ${MPI_PATH}/include)
MPI_PATH=/opt/homebrew

# --=== OPTIONAL User Variables ===--
# For build with matplotlibcpp:
# PYTHON_ENV_PATH : Path to conda `EllipticForest` environment directory
# NOTE: Also check ${HOME}/miniconda3 or ${HOME}/anaconda
PYTHON_ENV_PATH=${HOME}/miniforge3/envs/HydroForest

# PYTHON_VERSION : Version of Python in conda `EllipticForest` envrionment
PYTHON_VERSION=python3.9

# For pre-installed PETSc:
# PETSC_PATH : Path to PETSc install (i.e., ${PETSC_PATH}/include, ${PETSC_PATH}/lib, ...) 
PETSC_PATH=${HOME}/packages/petsc/petsc-build-mumps

# FORESTCLAW_PATH : Path to ForestClaw install (i.e., ${FORESTCLAW_PATH}/include, ${FORESTCLAW}/lib, ...)
FORESTCLAW_PATH=${HOME}/packages/forestclaw/forestclaw-build/local
# --=== END User Variables ===--

# --=== Create Build Directory ===--
BUILD_DIR=build-$(git branch --show-current)
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# --=== CMake Configure ===--
cmake ${ELLIPTIC_FOREST} \
    -DCMAKE_PREFIX_PATH=${PETSC_PATH} \
    -DCMAKE_INSTALL_PREFIX=local \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_C_COMPILER=mpicc \
    -DMPI_PATH=${MPI_PATH} \
    -DP4EST_PATH=${P4EST_PATH} \
    -DWITH_PETSC=true \
    -DPETSC_PATH=${PETSC_PATH} \
    -DWITH_MATPLOTLIBCPP=true \
    -DPYTHON_ENV_PATH=${PYTHON_ENV_PATH} \
    -DPYTHON_VERSION=${PYTHON_VERSION}

echo "Now cd to $(pwd)/${BUILD_DIR} and run make to compile"
