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

    message(FATAL_ERROR "PETSC_PATH was not set in configuration!")
    
endif()