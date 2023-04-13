## Matrix storage

The implementation exposes `Matrix` which stores the matrix as a 1D aligned storage where the size is known at compile time. `MatrixX` class addtionally allows for dynamic resizing.

## Static polymorphism with CRTP

`MatrixBase` implements the base class for which all computation kernels are implemented. This allows us to implement addtional matrix representations without having to re-implement computation kernels. This could also be achieved by virtual functions (and might look slightly cleaner), but with CRTP we avoid the cost of dynamic dispatch.

## Kernels

Higher performance is achieved by tiling the matrices such that smaller pieces of it fit in the higher levels of the cache. The implementation follows the strategies mentioned in [BLISlab](https://arxiv.org/pdf/1609.00076v1.pdf). The implementation also aims to break up the computation among multiple threads while avoiding false sharing.

## Multithreading

Kernels which utilize multithreading utilize a singleton global threadpool. The global threadpool already utilizes the maximum number of available threads and creation of additional threads will only serve to add overhead considering that the operations are compute intensive. Consequently, kernels are designed to be called in a serial manner.

**WARNING:**
Parallelizing calls to kernels may lead to additional delays due to inefficient waiting in the global threadpool and higher cache misses.
