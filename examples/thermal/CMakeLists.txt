project("Thermal")

add_executable(thermal
    main.cpp
)

target_include_directories(thermal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(thermal PUBLIC
    EllipticForest
)