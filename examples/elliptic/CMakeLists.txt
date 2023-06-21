project("Elliptic")

add_executable(elliptic
    main.cpp
)

target_include_directories(elliptic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(elliptic PUBLIC
    EllipticForest
)