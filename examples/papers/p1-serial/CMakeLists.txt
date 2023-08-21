project("P1Serial")

add_library(common-paper STATIC common.cpp)

set(PROBLEMS laplace-1 poisson-1 poisson-2 helmholtz-1 helmholtz-2 heat-1 allen-cahn-1)

foreach(problem IN LISTS PROBLEMS)
    add_executable(${problem} ${problem}.cpp)
    target_include_directories(${problem} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${problem} PUBLIC EllipticForest common-paper)
endforeach()

configure_file(run_all.sh run_all.sh)