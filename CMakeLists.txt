cmake_minimum_required(VERSION 3.10)
project(tBLAS VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "-O3 -Wall")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)

include(GNUInstallDirs)

# Include headers
include_directories(include/tBLAS)

# Create library
add_library(tBLAS INTERFACE)
target_include_directories(
  tBLAS
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(tBLAS INTERFACE Threads::Threads)

target_compile_features(tBLAS INTERFACE cxx_std_11)

enable_testing()
add_subdirectory(tests)