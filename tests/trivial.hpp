#ifndef TRIVIAL_HPP
#define TRIVIAL_HPP

#include <vector>
#include <cstddef>

namespace tBLAS_test
{

    /**
     * @brief Trivial implementation of matrix transpose and multiplication.
     *
     * Used to compare benchmark results.
     */
    class Trivial
    {
    public:
        /**
         * @brief Transpose a matrix.
         *
         * @tparam T The type of the matrix elements.
         * @param A The matrix to transpose.
         * @return std::vector<std::vector<T>> The transposed matrix.
         */
        template <typename T>
        static std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &A);

        /**
         * @brief Multiply two matrices.
         *
         * @tparam T The type of the matrix elements.
         * @param A The first matrix.
         * @param B The second matrix.
         * @return std::vector<std::vector<T>> The product of the two matrices.
         */
        template <typename T>
        static std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B);
    };

    /* ----------------------------- Implementation ----------------------------- */
    template <typename T>
    std::vector<std::vector<T>> Trivial::transpose(const std::vector<std::vector<T>> &A)
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
    std::vector<std::vector<T>> Trivial::matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B)
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

}; // namespace tBLAS_test

#endif // TRIVIAL_HPP