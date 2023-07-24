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
    Interpolation.cpp
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

if(WITH_PETSC)
    target_link_libraries(EllipticForest PUBLIC
        fishpack90 petsc sc p4est ${PYTHON_VERSION} lapack blas z
    )
else()
    target_link_libraries(EllipticForest PUBLIC
        fishpack90 sc p4est ${PYTHON_VERSION} lapack blas z
    )
endif()

install(FILES
    DataCache.hpp
    EllipticForest.hpp
    EllipticForestApp.hpp
    EllipticProblem.hpp
    FISHPACK.hpp
    GenericSingleton.hpp
    HPSAlgorithm.hpp
    Matrix.hpp
    P4est.hpp
    Patch.hpp
    PatchGrid.hpp
    PatchSolver.hpp
    PlotUtils.hpp
    Quadtree.hpp
    SpecialMatrices.hpp
    VTK.hpp
    Vector.hpp
    XMLTree.hpp
    DESTINATION include
)

install(TARGETS EllipticForest fishpack90
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)