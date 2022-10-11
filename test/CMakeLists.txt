include(CTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

file(GLOB tests "./*.cpp")
list(REMOVE_ITEM tests "./main.cpp")

foreach(file ${tests})
	set(name)
	get_filename_component(name ${file} NAME_WE)
	add_executable("${name}_tests" ${file} "./main.cpp")
    target_include_directories("${name}_tests" PUBLIC PROJECT_SOURCE_DIR/src)
	target_link_libraries("${name}_tests" PUBLIC EllipticForest gtest_main)
	gtest_discover_tests("${name}_tests")
endforeach()
