cmake_minimum_required(VERSION 3.10)
project(tBLAS VERSION 1.0)

include(GNUInstallDirs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -funroll-loops")

# Include headers
include_directories(include)

# Create library
add_library(tBLAS INTERFACE)
target_include_directories(
  tBLAS
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(tBLAS INTERFACE cxx_std_11)

enable_testing()
add_subdirectory(tests)