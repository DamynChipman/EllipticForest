project("WeakScaling")

add_executable(weak-scaling
    main.cpp
)

target_include_directories(weak-scaling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(weak-scaling PUBLIC
    EllipticForest
)