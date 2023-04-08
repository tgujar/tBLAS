cmake_minimum_required(VERSION 3.10)


# set(CMAKE_C_COMPILER /usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "-O3 -Wall")


project(tBLAS_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Locate Catch2 package
find_package(Catch2 3 REQUIRED)
find_package(Threads REQUIRED)

# Create an executable for the tests
add_executable(tests tests_threadpool.cpp tests_transpose.cpp tests_matmul.cpp tests_matrix.cpp)
add_executable(bench bench/bench.cpp)

# Link against the main library and Catch2
target_link_libraries(tests PRIVATE tBLAS Catch2::Catch2WithMain Threads::Threads)
target_link_libraries(bench PRIVATE tBLAS Catch2::Catch2WithMain Threads::Threads)


# Add tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(tests)