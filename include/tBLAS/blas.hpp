#ifndef BLAS_HPP
#define BLAS_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include "matrix.hpp"
#include "utils.hpp"
#include "./BLAS/matmul/matmul.hpp"
#include "./BLAS/transpose/transpose.hpp"

namespace tBLAS
{

    template <typename T, size_t M, size_t K>
    Matrix<T, K, M> transpose(const Matrix<T, M, K> &A, BLAS::transpose_kernel kernel = BLAS::transpose_kernel::xl);

    template <typename T>
    MatrixX<T> transpose(const MatrixX<T> &A, BLAS::transpose_kernel kernel = BLAS::transpose_kernel::xl);

    template <typename T, size_t M, size_t N, size_t K>
    Matrix<T, M, N> mutiply(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B, BLAS::matmul_kernel kernel = BLAS::matmul_kernel::sm);

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