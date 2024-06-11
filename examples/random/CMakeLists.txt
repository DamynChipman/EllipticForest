project("Random")

add_executable(random
    main.cpp
)

target_include_directories(random PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(random PUBLIC
    EllipticForest
)