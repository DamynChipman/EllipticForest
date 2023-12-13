add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

file(GLOB TESTS "./*.cpp")
list(REMOVE_ITEM TESTS "./main.cpp")

foreach(FILE ${TESTS})
	set(NAME)
	get_filename_component(NAME ${FILE} NAME_WE)
	add_executable("${NAME}_TESTS" ${FILE} "main.cpp")
    target_include_directories("${NAME}_TESTS" PUBLIC ${PROJECT_SOURCE_DIR}/src)
	target_link_libraries("${NAME}_TESTS" PUBLIC EllipticForest gtest_main)
	gtest_discover_tests("${NAME}_TESTS")
endforeach()

# Parallel unit testing
# add_subdirectory(mpi)