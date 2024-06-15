project("RefineCoarsen")

add_executable(refine-coarsen
    main.cpp
)

target_include_directories(refine-coarsen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(refine-coarsen PUBLIC
    EllipticForest
)