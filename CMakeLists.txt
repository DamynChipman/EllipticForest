cmake_minimum_required(VERSION 3.21)
project("EllipticForest" LANGUAGES CXX Fortran)
set(CMAKE_CXX_STANDARD 20)

# --=== External Pacakges ===--
add_subdirectory(external)

# --=== HydroForest Source Code ===--
add_subdirectory(src)

# --=== HydroForest Test Suite ===--
add_subdirectory(test)
enable_testing()

# --=== HydroForest Examples and Math598 Projects ===--
# add_subdirectory(examples)