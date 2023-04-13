cmake_minimum_required(VERSION 3.10)
project("tBLAS-example")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "-O3 -Wall")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)
find_package(tBLAS CONFIG REQUIRED)

add_executable(tBLAS-example src/main.cpp)
target_link_libraries(tBLAS-example tBLAS::tBLAS)