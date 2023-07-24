project("Heat")

add_executable(heat
    main.cpp
)

target_include_directories(heat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(heat PUBLIC
    EllipticForest
)