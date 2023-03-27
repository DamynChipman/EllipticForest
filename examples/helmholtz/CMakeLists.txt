project("Helmholtz")

add_executable(helmholtz
    main.cpp
)

target_include_directories(helmholtz PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(helmholtz PUBLIC
    EllipticForest
)