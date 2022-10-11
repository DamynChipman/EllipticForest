add_subdirectory(fishpack90)
add_subdirectory(matplotlibcpp)
add_subdirectory(petsc)
add_subdirectory(p4est)

FetchContent_GetProperties(matplotlibcpp)

if(
    ${FISHPACK90_FOUND} AND
    ${matplotlibcpp_POPULATED} AND
    ${PETSC_FOUND} AND
    ${P4EST_FOUND}
)

    set(EXTERNAL_PKGS_INCLUDE
        ${PYTHON_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIR}
        ${matplotlibcpp_SOURCE_DIR}
        ${PETSC_INCLUDE}
        ${P4EST_INCLUDE}
        CACHE INTERNAL "HydroForest external pacakges' include directories"
    )

    set(EXTERNAL_PKGS_LIB
        ${PYTHON_LIBRARY_DIR}
        ${PETSC_LIB}
        ${P4EST_LIB}
        CACHE INTERNAL "HydroForest external packages' library directories"
    )

    message(STATUS "External packages ready!")

else()

    message(STATUS "External packages not ready:")
    message(STATUS "FISHPACK90_FOUND = " ${FISHPACK90_FOUND})
    message(STATUS "matplotlibcpp_POPULATED = " ${matplotlibcpp_POPULATED})
    message(STATUS "PETSC_FOUND = " ${PETSC_FOUND})
    message(STATUS "P4EST_FOUND = " ${P4EST_FOUND})
    message(FATAL_ERROR "External packages not ready...")

endif()