project("Elliptic-Multiple")

add_executable(elliptic-multiple
    main.cpp
)

target_include_directories(elliptic-multiple PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(elliptic-multiple PUBLIC
    EllipticForest
)