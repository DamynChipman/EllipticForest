project("Toybox")

add_executable(toybox
    main.cpp
)

target_include_directories(toybox PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(toybox PUBLIC
    EllipticForest
)

configure_file(options.ini options.ini)