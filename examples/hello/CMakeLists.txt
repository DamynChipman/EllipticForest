project("Hello")

add_executable(hello
    main.cpp
)

target_include_directories(hello PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hello PUBLIC
    EllipticForest
)