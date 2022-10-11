# --=== User Variables ===--
# HOME : Path to home directory
HOME=/Users/damynchipman

# PYTHON_ENV_PATH : Path to conda `HydroForest` environment directory
# NOTE: Also check ${HOME}/miniconda3 or ${HOME}/anaconda
PYTHON_ENV_PATH=${HOME}/miniforge3/envs/HydroForest

# PYTHON_VERSION : Version of Python in conda `HydroForest` envrionment
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