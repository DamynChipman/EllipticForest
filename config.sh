#!/bin/sh

# --=== Path Variables ===--
# HOME : Path to home directory
HOME=/Users/damynchipman

# MPI_PATH : Directory with MPI headers (i.e. ${MPI_PATH}/include)
MPI_PATH=/opt/homebrew

# P4EST_PATH : Path to p4est install (i.e., ${P4EST_PATH}/include, ${P4EST_PATH}/lib, ...)
P4EST_PATH=${HOME}/packages/p4est/p4est/build/local

# PETSC_PATH : Path to PETSc install (i.e., ${PETSC_PATH}/include, ${PETSC_PATH}/lib, ...) 
PETSC_PATH=${HOME}/packages/petsc/petsc-build-mumps

# BUILD_DIR : Build directory
BUILD_DIR=build-$(git branch --show-current)

# --=== CMake Configure ===--
cmake -S . -B ${BUILD_DIR} \
    -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/local \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_C_COMPILER=mpicc \
    -DMPI_PATH=${MPI_PATH} \
    -DP4EST_PATH=${P4EST_PATH} \
    -DPETSC_PATH=${PETSC_PATH}
