# --=== External Packages ===--
if (${matplotlibcpp})
    FetchContent_GetProperties(matplotlibcpp)
endif()

find_package(Kokkos REQUIRED)
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
endif()

# --=== Create EllipticForest Library ===--
add_library(EllipticForest)
set_target_properties(EllipticForest PROPERTIES EXPORT_NAME EllipticForest)

target_link_libraries(EllipticForest PUBLIC common)

set(ELLIPTIC_FOREST_SOURCES
    XMLTree.cpp
    EllipticForestApp.cpp
    Interpolation.cpp
    FISHPACK.cpp
    MPI.cpp
    P4est.cpp
    PETSc.cpp
    VTK.cpp
)

target_sources(EllipticForest PRIVATE
    ${ELLIPTIC_FOREST_SOURCES}
)

target_include_directories(EllipticForest PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXTERNAL_PKGS_INCLUDE}
)

set(${PROJECT_NAME}_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} ${EXTERNAL_PKGS_INCLUDE} CACHE INTERNAL "EllipticForest's include directories" FORCE)

target_link_directories(EllipticForest PUBLIC
    ${EXTERNAL_PKGS_LIB}
)

if(WITH_PETSC)
    target_link_libraries(EllipticForest PUBLIC
        fishpack90 Kokkos::kokkos petsc sc p4est ${PYTHON_VERSION} lapack blas z
    )
else()
    target_link_libraries(EllipticForest PUBLIC
        fishpack90 Kokkos::kokkos sc p4est ${PYTHON_VERSION} lapack blas z
    )
endif()

install(
    FILES
    DataCache.hpp
    EllipticForest.hpp
    EllipticForestApp.hpp
    EllipticProblem.hpp
    FISHPACK.hpp
    ForestClaw.hpp
    GenericSingleton.hpp
    HPSAlgorithm.hpp
    Interpolation.hpp
    MPI.hpp
    Matrix.hpp
    Mesh.hpp
    P4est.hpp
    PETSc.hpp
    Patch.hpp
    PatchGrid.hpp
    PatchSolver.hpp
    PlotUtils.hpp
    QuadNode.hpp
    Quadtree.hpp
    SpecialMatrices.hpp
    VTK.hpp
    Vector.hpp
    VectorBase.hpp
    XMLTree.hpp
    DESTINATION include
)

install(TARGETS EllipticForest fishpack90
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)