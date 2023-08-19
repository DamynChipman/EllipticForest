project("Elliptic-Single")

add_executable(elliptic-single
    main.cpp
)

target_include_directories(elliptic-single PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(elliptic-single PUBLIC
    EllipticForest
)