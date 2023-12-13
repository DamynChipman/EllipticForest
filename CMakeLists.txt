# ====================================================================================================
# EllipticForest Top-Level CMakeLists.txt
# ====================================================================================================
cmake_minimum_required(VERSION 3.21)
project(
    "EllipticForest"
    LANGUAGES CXX C Fortran
)
set(CMAKE_CXX_STANDARD 20)

# Create common library for properties and project wide compiler stuff
add_library(common INTERFACE)

# ====================================================================================================
# External packages
# ====================================================================================================
# Perform the submodule update
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# --=== MPI ===--
set(MPI_PATH "" CACHE STRING "Path to user MPI installation")
if(MPI_PATH STREQUAL "")
    message(FATAL_ERROR "MPI_PATH not provided but required.")
endif()

# --=== LAPACK and BLAS ===--
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# --=== FISHPACK90 ===--
set(FISHPACK90_PATH "" CACHE STRING "Path to user FISHPACK90 installation")
if(FISHPACK90_PATH STREQUAL "")

    message(STATUS "FISHPACK90_PATH not provided; will build FISHPACK90 via git submodule.")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/fishpack90/CMakeLists.txt")
        message(FATAL_ERROR "The submodule fishpack90 was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
    endif()
    set(CMAKE_Fortran_FLAGS "-w -fallow-argument-mismatch -fdefault-real-8")
    add_subdirectory(extern/fishpack90)
    set(FISHPACK90_LIBRARIES "fishpack90" CACHE INTERNAL "FISHPACK90 libraries")

else()

    message(STATUS "Pre-installed FISHPACK90 path provided: FISHPACK90_PATH = " ${FISHPACK90_PATH})
    if(EXISTS ${FISHPACK90_PATH}/src/fish.f)
        set(FISHPACK_INCLUDE ${FISHPACK90_PATH}/src CACHE INTERNAL "FISHPACK90 include directory")
    elseif(EXISTS ${FISHPACK90_PATH}/include/fish.f)
        set(FISHPACK_INCLUDE ${FISHPACK90_PATH}/include CACHE INTERNAL "FISHPACK90 include directory")
    endif()

    if(EXISTS ${FISHPACK90_PATH}/lib/libfishpack90.dylib)
        set(FISHPACK90_LIB ${FISHPACK90_PATH}/lib CACHE INTERNAL "FISHPACK90 library directory")
    elseif(EXISTS ${FISHPACK90_PATH}/lib/libfishpack90.a)
        set(FISHPACK90_LIB ${FISHPACK90_PATH}/lib CACHE INTERNAL "FISHPACK90 library directory")
    elseif(EXISTS ${FISHPACK90_PATH}/lib/libfishpack90.so)
        set(FISHPACK90_LIB ${FISHPACK90_PATH}/lib CACHE INTERNAL "FISHPACK90 library directory")
    else()
        message(FATAL_ERROR "Could not find libfishpack90")
    endif()

    set(FISHPACK90_LIBRARIES "fishpack90" CACHE INTERNAL "FISHPACK90 libraries")
    set(FISHPACK90_FOUND TRUE CACHE INTERNAL "Flag for if FISHPACK90 was found")

    message(STATUS "Found external FISHPACK90:")
    message(STATUS "  FISHPACK90_INCLUDE = " ${FISHPACK90_INCLUDE})
    message(STATUS "  FISHPACK90_LIB = " ${FISHPACK90_LIB})

    if (${FISHPACK90_FOUND})
        list(APPEND LIST_EXTERNAL_PKGS_INCLUDE ${FISHPACK90_INCLUDE})
        list(APPEND LIST_EXTERNAL_PKGS_LIB ${FISHPACK90_LIB})
    else()
        message(STATUS "Required external package `FISHPACK90` not ready:")
        message(STATUS "  FISHPACK90_FOUND = " ${FISHPACK90_FOUND})
        message(FATAL_ERROR "Required external package `FISHPACK90` not ready...")
    endif()

endif()

# --=== p4est ===--
set(P4EST_PATH "" CACHE STRING "Path to user p4est installation")
if(P4EST_PATH STREQUAL "")

    message(STATUS "P4EST_PATH not provided; will build p4est via git submodule.")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/p4est/CMakeLists.txt")
        message(FATAL_ERROR "The submodule p4est was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
    endif()
    
    set(mpi ON)
    set(BUILD_TESTING FALSE) # No need to build testing for p4est; enabled by default in p4est
    add_subdirectory(extern/p4est)
    set(BUILD_TESTING TRUE)
    set(P4EST_LIBRARIES "p4est" CACHE INTERNAL "p4est libraries")

else()

    message(STATUS "Pre-installed p4est path provided: P4EST_PATH = " ${P4EST_PATH})
    if(EXISTS ${P4EST_PATH}/include/p4est.h)
        set(P4EST_INCLUDE ${P4EST_PATH}/include CACHE INTERNAL "p4est include directory")
    else()
        message(FATAL_ERROR "Could not find p4est.h")
    endif()

    if(EXISTS ${P4EST_PATH}/lib/libp4est.dylib)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    elseif(EXISTS ${P4EST_PATH}/lib/libp4est.a)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    elseif(EXISTS ${P4EST_PATH}/lib/libp4est.so)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    else()
        message(FATAL_ERROR "Could not find libp4est")
    endif()

    set(P4EST_LIBRARIES "sc" "p4est" "z" CACHE INTERNAL "p4est libraries")
    set(P4EST_FOUND TRUE CACHE INTERNAL "Flag for if p4est was found")

    message(STATUS "Found external p4est:")
    message(STATUS "  P4EST_INCLUDE = " ${P4EST_INCLUDE})
    message(STATUS "  P4EST_LIB = " ${P4EST_LIB})

    if (${P4EST_FOUND})
        list(APPEND LIST_EXTERNAL_PKGS_INCLUDE ${P4EST_INCLUDE})
        list(APPEND LIST_EXTERNAL_PKGS_LIB ${P4EST_LIB})
    else()
        message(STATUS "Required external package `p4est` not ready:")
        message(STATUS "  P4EST_FOUND = " ${P4EST_FOUND})
        message(FATAL_ERROR "Required external package `p4est` not ready...")
    endif()
    
endif()

# --=== PETSc ===--
set(PETSC_PATH "" CACHE STRING "Path to user PETSc installation")
if(PETSC_PATH STREQUAL "")

    message(STATUS "PETSC_PATH was not provided; will build PETSc via ExternalProject.")
    ExternalProject_Add(
        petsc-external
        GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
        GIT_TAG release
        GIT_PROGRESS true
        PREFIX ${CMAKE_BINARY_DIR}/extern/petsc-src
        BUILD_IN_SOURCE true
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/extern/petsc-build --with-mpi-dir=${MPI_PATH} --with-fc=0
        UPDATE_COMMAND ""
    )

    set(PETSC_INCLUDE ${CMAKE_BINARY_DIR}/extern/petsc-build/include CACHE INTERNAL "PETSc include directory")
    set(PETSC_LIB ${CMAKE_BINARY_DIR}/extern/petsc-build/lib CACHE INTERNAL "PETSc library directory")
    set(PETSC_LIBRARIES "petsc" CACHE INTERNAL "PETSc libraries")
    set(PETSC_FOUND TRUE CACHE INTERNAL "Flag for if PETSc was found")

else()

    message(STATUS "Pre-installed PETSc path provided: PETSC_PATH = " ${PETSC_PATH})
    if(EXISTS "${PETSC_PATH}/include/petsc.h")
        set(PETSC_INCLUDE ${PETSC_PATH}/include CACHE INTERNAL "PETSc include directory")
    else()
        message(FATAL_ERROR "Could not find petsc.h")
    endif()

    if(EXISTS ${PETSC_PATH}/lib/libpetsc.dylib)
        set(PETSC_LIB ${PETSC_PATH}/lib CACHE INTERNAL "PETSc library directory")
    elseif(EXISTS ${PETSC_PATH}/lib/libpetsc.a)
        set(PETSC_LIB ${PETSC_PATH}/lib CACHE INTERNAL "PETSc library directory")
    elseif(EXISTS ${PETSC_PATH}/lib/libpetsc.so)
        set(PETSC_LIB ${PETSC_PATH}/lib CACHE INTERNAL "PETSc library directory")
    else()
        message(FATAL_ERROR "Could not find libpetsc")
    endif()

    set(PETSC_LIBRARIES "petsc" CACHE INTERNAL "PETSc libraries")
    set(PETSC_FOUND TRUE CACHE INTERNAL "Flag for if PETSc was found")

    message(STATUS "Found external PETSc:")
    message(STATUS "  PETSC_INCLUDE = " ${PETSC_INCLUDE})
    message(STATUS "  PETSC_LIB = " ${PETSC_LIB})

    if (${PETSC_FOUND})
        list(APPEND LIST_EXTERNAL_PKGS_INCLUDE ${PETSC_INCLUDE})
        list(APPEND LIST_EXTERNAL_PKGS_LIB ${PETSC_LIB})
        target_compile_definitions(common INTERFACE USE_PETSC)
    else()
        message(STATUS "Required external package `petsc` not ready:")
        message(STATUS "  PETSC_FOUND = " ${PETSC_FOUND})
        message(FATAL_ERROR "Required external package `petsc` not ready...")
    endif()

endif()

# Set up external package include and library directories
set(EXTERNAL_PKGS_INCLUDE ${LIST_EXTERNAL_PKGS_INCLUDE} CACHE INTERNAL "EllipticForest external packages' include directories")
set(EXTERNAL_PKGS_LIB ${LIST_EXTERNAL_PKGS_LIB} CACHE INTERNAL "EllipticForest external packages' library directories")

# ====================================================================================================
# EllipticForest Source Code
# ====================================================================================================
add_subdirectory(src)

# ====================================================================================================
# EllipticForest Testing
# ====================================================================================================
if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()

# ====================================================================================================
# EllipticForest Example Apps
# ====================================================================================================
add_subdirectory(examples)