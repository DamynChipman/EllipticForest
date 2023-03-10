add_executable(forestclaw_hello main.cpp)
target_include_directories(forestclaw_hello PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EXTERNAL_PKGS_INCLUDE})
target_link_directories(forestclaw_hello PUBLIC ${EXTERNAL_PKGS_LIB})
target_link_libraries(forestclaw_hello PUBLIC z sc p4est petsc EllipticForest forestclaw)
