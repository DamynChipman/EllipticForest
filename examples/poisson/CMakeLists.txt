project("Poisson")

add_executable(poisson
    main.cpp
)

target_include_directories(poisson PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(poisson PUBLIC
    EllipticForest
)