# tBLAS

A header only multithreaded and cache-optimized linear algebra library

### TODO and future

- **Optimized kernel for transpose of smaller matrices**: The current implementation supports fast transpose using tiling and multithreading for large matrices. However, for smaller matrices the setup of the matrices involves larger overhead. Smaller matrices can be optimized similar to larger one's, with different tiling parameters. The current implementation does a naive implementation for smaller matrices which might fit into L1 cache.

- **Tuning params**: Presently the parameters for block tiling the matrices are defined in `constants.hpp`. These values are not tuned and should be interpreted as sensible defaults. In an ideal scenario these would be tuned for a given CPU with known cache sizes. The current implementation can be improved by allowing configuration by passing preprocessor definitions so the user of the library can input the cache sizes and the block sizes can then be derived from these by using `constexpr` based on some heuristics.

- **More tests**: The current tests showcase the correctness of the solution fairly well. There is still room for more coverage.

### Installation

From project root run

```
$ cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/your/installation/path
$ cmake --build . --config Release --target install -- -j $(nproc)
```

## Running tests

From build directory run

```
$ cmake ..
$ make tests
$ ctest
```

### Usage

Example CMake snippet

```
project("tBLAS-example")

find_package(tBLAS CONFIG REQUIRED)

add_executable(tBLAS-example src/main.cpp)
target_link_libraries(tBLAS-example tBLAS::tBLAS)
```
