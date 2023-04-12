#ifndef BLAS_HPP
#define BLAS_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include "matrix.hpp"
#include "./BLAS/matmul/matmul.hpp"
#include "./BLAS/transpose/transpose.hpp"

namespace tBLAS
{
    /**
     * @brief Performs out of place transpose a matrix of size MxK
     *
     * @tparam T Type of the matrix
     * @tparam M Number of rows
     * @tparam K Number of columns
     * @param A Input matrix
     * @param kernel Kernel to use, depends on the size of the matrix
     * @return Matrix<T, K, M> Transposed matrix
     */
    template <typename T, size_t M, size_t K>
    Matrix<T, K, M> transpose(const Matrix<T, M, K> &A, BLAS::transpose_kernel kernel = BLAS::transpose_kernel::xl);

    /**
     * @brief Performs out of place transpose a matrix of dynamic size
     *
     * @tparam T Type of the matrix
     * @param A Input matrix
     * @param kernel Kernel to use, depends on the size of the matrix
     * @return MatrixX<T> Transposed matrix
     */
    template <typename T>
    MatrixX<T> transpose(const MatrixX<T> &A, BLAS::transpose_kernel kernel = BLAS::transpose_kernel::xl);

    /**
     * @brief Performs matrix multiplication of two matrices of size MxK and KxN
     *
     * By default will choose the kernel for small matrices which might fit into L1 cache.
     *
     * @tparam T Type of the matrix
     * @tparam M Number of rows of A
     * @tparam N Number of columns of B
     * @tparam K Number of columns of A and rows of B
     * @param A Left matrix
     * @param B Right matrix
     * @param kernel Kernel to use, depends on the size of the matrix
     * @return Matrix<T, M, N> Result matrix
     */
    template <typename T, size_t M, size_t N, size_t K>
    Matrix<T, M, N> multiply(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B, BLAS::matmul_kernel kernel = BLAS::matmul_kernel::sm);

    /**
     * @brief Performs matrix multiplication of two matrices of dynamic size
     *
     * By default will choose the kernel for large matrices which might not fit into L1+L2 cache.
     *
     * @tparam T Type of the matrix
     * @param A Left matrix
     * @param B Right matrix
     * @param kernel Kernel to use, depends on the size of the matrix
     * @return MatrixX<T> Result matrix
     */
    template <typename T>
    MatrixX<T> multiply(const MatrixX<T> &A, const MatrixX<T> &B, BLAS::matmul_kernel kernel = BLAS::matmul_kernel::xl);

    /* ----------------------------- Implementation ----------------------------- */

    template <typename T, size_t M, size_t K>
    Matrix<T, K, M> transpose(const Matrix<T, M, K> &A, BLAS::transpose_kernel kernel)
    {
        Matrix<T, K, M> C;
        BLAS::matrix_transpose(A, C, kernel);
        return C;
    }

    template <typename T>
    MatrixX<T> transpose(const MatrixX<T> &A, BLAS::transpose_kernel kernel)
    {
        MatrixX<T> C(A.cols(), A.rows());
        BLAS::matrix_transpose(A, C, kernel);
        return C;
    }

    template <typename T>
    MatrixX<T> multiply(const MatrixX<T> &A, const MatrixX<T> &B, BLAS::matmul_kernel kernel)
    {
        MatrixX<T> C(A.rows(), B.cols());
        BLAS::matmul(A, B, C, kernel);
        return C;
    }

    template <typename T, size_t M, size_t N, size_t K>
    Matrix<T, M, N> multiply(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B, BLAS::matmul_kernel kernel)
    {
        Matrix<T, M, N> C;
        BLAS::matmul(A, B, C, kernel);
        return C;
    }

}; // namespace tBLAS

#endif // BLAS_HPP