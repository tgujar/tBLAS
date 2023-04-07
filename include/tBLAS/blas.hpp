#ifndef BLAS_HPP
#define BLAS_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include "matrix.hpp"
#include "constants.hpp"
#include "utils.hpp"

namespace tBLAS
{

    class BLAS
    {
    public:
        template <typename T>
        static std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &A);

        template <typename T>
        static std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B);

        template <typename T, size_t M, size_t N>
        static std::array<std::array<T, M>, N> transpose(const std::array<std::array<T, M>, N> &A);

        template <typename T, size_t M, size_t N>
        static std::array<std::array<T, M>, N> matmul(const std::array<std::array<T, M>, N> &A, const std::array<std::array<T, M>, N> &B);

        template <typename T, size_t M, size_t N>
        static tBLAS::Matrix<T, M, N> transpose(const tBLAS::Matrix<T, M, N> &A);

        template <typename T, size_t M, size_t N, size_t K>
        static tBLAS::Matrix<T, M, N> matmul(const tBLAS::Matrix<T, M, K> &A, const tBLAS::Matrix<T, K, N> &B);

        template <typename T>
        static tBLAS::MatrixX<T> transpose(const tBLAS::MatrixX<T> &A);

        template <typename T>
        static tBLAS::MatrixX<T> matmul(const tBLAS::MatrixX<T> &A, const tBLAS::MatrixX<T> &B);
    };

    template <typename T>
    std::vector<std::vector<T>> BLAS::transpose(const std::vector<std::vector<T>> &A)
    {
        std::vector<std::vector<T>> B(A[0].size(), std::vector<T>(A.size()));
        for (int i = 0; i < A.size(); ++i)
        {
            for (int j = 0; j < A[0].size(); ++j)
            {
                B[j][i] = A[i][j];
            }
        }
        return B;
    }

    template <typename T>
    std::vector<std::vector<T>> BLAS::matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B)
    {
        std::vector<std::vector<T>> C(A.size(), std::vector<T>(B[0].size()));

        for (int i = 0; i < A.size(); ++i)
        {
            for (int j = 0; j < B[0].size(); ++j)
            {
                for (int k = 0; k < B.size(); ++k)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // TODO: add tests for this
    template <typename T, size_t M, size_t N>
    std::array<std::array<T, M>, N> transpose(const std::array<std::array<T, M>, N> &A)
    {
        std::array<std::array<T, M>, N> B;
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                B[j][i] = A[i][j];
            }
        }
        return B;
    }

    // TODO: add tests for this
    template <typename T, size_t M, size_t N>
    std::array<std::array<T, M>, N> matmul(const std::array<std::array<T, M>, N> &A, const std::array<std::array<T, M>, N> &B)
    {
        std::array<std::array<T, M>, N> C;
        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < M; ++k)
            {
                for (int j = 0; j < M; ++j)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // template <typename T, size_t M, size_t N>
    // static tBLAS::Matrix<T, M, N> transpose(const tBLAS::Matrix<T, M, N> &A);

    // template <typename T>
    // static tBLAS::MatrixX<T> transpose(const tBLAS::MatrixX<T> &A);

    template <typename T>
    MatrixX<T> BLAS::matmul(const tBLAS::MatrixX<T> &A, const tBLAS::MatrixX<T> &B)
    {
        MatrixX<T> C(A.rows(), B.cols());
        matrix_gemm(A, B, C);
        return C;
    }

    template <typename T, size_t M, size_t N, size_t K>
    Matrix<T, M, N> BLAS::matmul(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B)
    {
        Matrix<T, M, N> C;
        matrix_gemm(A, B, C);
        return C;
    }

}; // namespace tBLAS

#endif // BLAS_HPP