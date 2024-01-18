project("AllenCahn")

add_executable(allen-cahn
    main.cpp
)

target_include_directories(allen-cahn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(allen-cahn PUBLIC
    EllipticForest
)