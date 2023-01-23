option(MPI_PATH "Path to MPI headers (i.e. MPI_PATH/include)" "")
if(PETSC_PATH)

    if(EXISTS ${PETSC_PATH}/include/petsc.h)
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

    set(PETSC_FOUND TRUE CACHE INTERNAL "Flag for if PETSc was found")

    message(STATUS "Found external PETSc:")
    message(STATUS "  PETSC_INCLUDE = " ${PETSC_INCLUDE})
    message(STATUS "  PETSC_LIB = " ${PETSC_LIB})

else()

    message(STATUS "PETSC_PATH was not set in configuration, building petsc from scratch...")
    
    include(ExternalProject)
    ExternalProject_Add(
        petsc-download
        GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
        GIT_TAG release
        GIT_PROGRESS true
        PREFIX ${CMAKE_BINARY_DIR}/_deps/petsc-src
        BUILD_IN_SOURCE true
        CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/_deps/petsc-build --with-mpi-dir=${MPI_PATH} --download-zlib --with-p4est-dir=${P4EST_PATH} --with-p4est=1
        BUILD_COMMAND make
        INSTALL_COMMAND make install
    )

    set(PETSC_INCLUDE ${CMAKE_BINARY_DIR}/_deps/petsc-build/include CACHE INTERNAL "PETSc include directory")
    set(PETSC_LIB ${CMAKE_BINARY_DIR}/_deps/petsc-build/lib CACHE INTERNAL "PETSc library directory")
    set(PETSC_FOUND TRUE CACHE INTERNAL "Flag for if PETSc was found")
    
endif()