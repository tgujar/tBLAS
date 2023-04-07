#ifndef BLAS_HPP
#define BLAS_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include "matrix.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include "./BLAS/matmul.hpp"
#include "./BLAS/transpose.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        template <typename T, size_t M, size_t K>
        tBLAS::Matrix<T, K, M> transpose(const tBLAS::Matrix<T, M, K> &A);

        template <typename T>
        tBLAS::MatrixX<T> transpose(const tBLAS::MatrixX<T> &A);

        template <typename T, size_t M, size_t N, size_t K>
        tBLAS::Matrix<T, M, N> matmul(const tBLAS::Matrix<T, M, K> &A, const tBLAS::Matrix<T, K, N> &B);

        template <typename T>
        tBLAS::MatrixX<T> matmul(const tBLAS::MatrixX<T> &A, const tBLAS::MatrixX<T> &B);
    };

    template <typename T, size_t M, size_t K>
    Matrix<T, K, M> BLAS::transpose(const tBLAS::Matrix<T, M, K> &A)
    {
        Matrix<T, K, M> C;
        BLAS::matrix_transpose(A, C);
        return C;
    }

    template <typename T>
    MatrixX<T> BLAS::transpose(const tBLAS::MatrixX<T> &A)
    {
        MatrixX<T> C(A.cols(), A.rows());
        BLAS::matrix_transpose(A, C);
        return C;
    }

    template <typename T>
    MatrixX<T> BLAS::matmul(const tBLAS::MatrixX<T> &A, const tBLAS::MatrixX<T> &B)
    {
        MatrixX<T> C(A.rows(), B.cols());
        BLAS::matrix_gemm(A, B, C);
        return C;
    }

    template <typename T, size_t M, size_t N, size_t K>
    Matrix<T, M, N> BLAS::matmul(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B)
    {
        Matrix<T, M, N> C;
        BLAS::matrix_gemm(A, B, C);
        return C;
    }

}; // namespace tBLAS

#endif // BLAS_HPP