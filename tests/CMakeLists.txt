cmake_minimum_required(VERSION 3.10)

# Locate Catch2 package
find_package(Catch2 3 REQUIRED)

# Create an executable for the tests
add_executable(tests tests_threadpool.cpp tests_transpose.cpp tests_matmul.cpp tests_matrix.cpp)
add_executable(bench bench/bench.cpp)

# Link against the main library and Catch2
target_link_libraries(tests PRIVATE tBLAS Catch2::Catch2WithMain)
target_link_libraries(bench PRIVATE tBLAS Catch2::Catch2WithMain)

# Add tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(tests)