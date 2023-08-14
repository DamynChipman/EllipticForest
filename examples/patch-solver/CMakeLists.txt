project("Patch-Solver")

add_executable(patch-solver
    main.cpp
)

target_include_directories(patch-solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(patch-solver PUBLIC
    EllipticForest
)