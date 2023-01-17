cmake_minimum_required(VERSION 3.21)
project("EllipticForest" LANGUAGES CXX Fortran)
set(CMAKE_CXX_STANDARD 20)

# --=== External Pacakges ===--
add_subdirectory(external)

# --=== EllipticForest Source Code ===--
add_subdirectory(src)

# --=== EllipticForest Test Suite ===--
add_subdirectory(test)
enable_testing()

# --=== EllipticForest Examples ===--
add_subdirectory(examples)