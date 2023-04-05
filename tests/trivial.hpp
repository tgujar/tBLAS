#ifndef TRIVIAL_HPP
#define TRIVIAL_HPP

#include <vector>
#include <cstddef>

namespace tBLAS_test
{

    /*
     * Trivial implementation of matrix transpose and matrix multiplication.
     */
    class Trivial
    {
    public:
        template <typename T>
        static std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &A)
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
        static std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B)
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

        template <typename T, std::size_t M, std::size_t N>
        static std::array<std::array<T, N>, M> transpose(const std::array<std::array<T, M>, N> &A)
        {
            std::array<std::array<T, N>, M> B{};
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    B[j][i] = A[i][j];
                }
            }
            return B;
        }

        template <typename T, std::size_t M, std::size_t N, std::size_t K>
        static std::array<std::array<T, N>, M> matmul(const std::array<std::array<T, K>, M> &A, const std::array<std::array<T, N>, K> &B)
        {
            std::array<std::array<T, N>, M> C{};
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    for (int k = 0; k < K; ++k)
                    {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }
    };

}; // namespace tBLAS_test

#endif // TRIVIAL_HPP