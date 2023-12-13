# --=== External Packages ===--
if (${matplotlibcpp})
    FetchContent_GetProperties(matplotlibcpp)
endif()

# --=== Create EllipticForest Library ===--
add_library(EllipticForest)
set_target_properties(EllipticForest PROPERTIES EXPORT_NAME EllipticForest)
target_link_libraries(EllipticForest PUBLIC common)

# EllipticForest sources
file(GLOB ELLIPTIC_FOREST_SOURCES "*.cpp")
file(GLOB ELLIPTIC_FOREST_HEADERS "*.hpp")
target_sources(EllipticForest PUBLIC ${ELLIPTIC_FOREST_SOURCES} ${ELLIPTIC_FOREST_HEADERS})
set_target_properties(EllipticForest PROPERTIES PUBLIC_HEADER "${ELLIPTIC_FOREST_HEADERS}")

# FiniteVolume Patches
add_subdirectory(Patches/FiniteVolume)

target_include_directories(EllipticForest PUBLIC
    ${P4EST_INCLUDE}
    ${PETSC_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(EllipticForest PUBLIC
    ${P4EST_LIB}
    ${PETSC_LIB}
)

target_link_libraries(EllipticForest PUBLIC
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${FISHPACK90_LIBRARIES}
    ${P4EST_LIBRARIES}
    ${PETSC_LIBRARIES}
    ${PYTHON_VERSION}
)

install(
    DIRECTORY
        Patches
    DESTINATION
        include
    FILES_MATCHING PATTERN
        "*.hpp"
)

install(
    TARGETS
        EllipticForest
    EXPORT
        EllipticForest-targets
    ARCHIVE DESTINATION
        lib
    LIBRARY DESTINATION
        lib
    RUNTIME DESTINATION
        bin
    INCLUDES DESTINATION
        include
    PUBLIC_HEADER DESTINATION
        include
)