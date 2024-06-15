project("PressurePoisson")

add_executable(pressure-poisson
    main.cpp
)

target_include_directories(pressure-poisson PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pressure-poisson PUBLIC
    EllipticForest
)