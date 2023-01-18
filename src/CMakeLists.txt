# --=== External Packages ===--
if (${matplotlibcpp})
    FetchContent_GetProperties(matplotlibcpp)
endif()

# --=== Create EllipticForest Library ===--
add_library(EllipticForest)
set_target_properties(EllipticForest PROPERTIES EXPORT_NAME EllipticForest)

target_link_libraries(EllipticForest PUBLIC common)

set(ELLIPTIC_FOREST_SOURCES
    EllipticForestApp.cpp
    FISHPACK.cpp
    P4est.cpp
    VTK.cpp
)

target_sources(EllipticForest PRIVATE
    ${ELLIPTIC_FOREST_SOURCES}
)

target_include_directories(EllipticForest PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXTERNAL_PKGS_INCLUDE}
)

target_link_directories(EllipticForest PUBLIC
    ${EXTERNAL_PKGS_LIB}
)

target_link_libraries(EllipticForest PUBLIC
    fishpack90 petsc sc p4est ${PYTHON_VERSION} lapack blas
)