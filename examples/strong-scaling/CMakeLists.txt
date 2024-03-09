project("StrongScaling")

add_executable(strong-scaling
    main.cpp
)

target_include_directories(strong-scaling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(strong-scaling PUBLIC
    EllipticForest
)