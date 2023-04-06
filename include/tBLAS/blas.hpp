#ifndef BLAS_HPP
#define BLAS_HPP

#include <vector>
#include <array>
#include "matrix.hpp"
#include "constants.hpp"
#include "utils.hpp"

namespace tBLAS
{
    template <typename T>
    using VPANEL = MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>;
    template <typename T>
    using HPANEL = MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>;

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
        static tBLAS::MatrixXd<T, M, N> transpose(const tBLAS::MatrixXd<T, M, N> &A);

        template <typename T, size_t M, size_t N>
        static tBLAS::MatrixXd<T, M, N> matmul(const tBLAS::MatrixXd<T, M, N> &A, const tBLAS::MatrixXd<T, M, N> &B);
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
    static std::array<std::array<T, M>, N> transpose(const std::array<std::array<T, M>, N> &A)
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
    static std::array<std::array<T, M>, N> matmul(const std::array<std::array<T, M>, N> &A, const std::array<std::array<T, M>, N> &B)
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
    // static tBLAS::MatrixXd<T, M, N> transpose(const tBLAS::MatrixXd<T, M, N> &A);

    template <typename T, size_t M, size_t N, size_t K>
    static MatrixXd<T, M, N> matmul(const MatrixXd<T, M, K> &A, const MatrixXd<T, K, N> &B)
    {
        MatrixXd<T, M, N> C;
        MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS> packA;
        MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS> packB;
        for (size_t i_mc = 0; i_mc < M; i_mc += KERNEL_MC)
        {
            size_t mc = min(M - i_mc, KERNEL_MC);
            for (size_t i_kc = 0; i_kc < K; i_kc += KERNEL_KC)
            {
                size_t kc = min(K - i_kc, KERNEL_KC);
                for (size_t i_mr = 0; i_mr < mc; i_mr += KERNEL_MR)
                {
                    pack_vertical(A.cbegin() + i_kc + K * (i_mc + i_mr), packA.begin() + i_mr * kc, {min(mc - i_mr, KERNEL_MR), kc})
                }
            }

            for (size_t i_nc = 0; i_nc < N; i_nc += KERNEL_NC)
            {
                size_t nc = min(n - i_nc, DGEMM_NC);
                for (size_t i_nr = 0; i_nr < nc; i_nr += KERNEL_NR)
                {
                    pack_horizontal(B.cbegin() + i_kc + K * (i_nc + i_nr), packB.begin() + i_nr * kc, {kc, min(nc - i_nr, KERNEL_NR)})
                }
            }

            macro_kernel_gemm<T, M, N>(mc, nc, kc, packA, packB, C.begin() + i_mc * N + i_nc);
        }
        return C;
    }

    template <typename T, size_t M, size_t N>
    static void macro_kernel_gemm(
        size_t mc, size_t nc, size_t kc,
        const VPANEL<T> &packA,
        const HPANEL<T> &packB,
        typename tBLAS::MatrixXd<T, M, N>::iterator &C_itr)
    {
        for (size_t i = 0; i < mc; i += KERNEL_MR)
        {
            for (size_t j = 0; j < nc; j += KERNEL_NR)
            {
                micro_kernel_gemm<T, M, N>(
                    kc,
                    min(mc - i, KERNEL_MR),
                    min(nc - j, KERNEL_NR),
                    packA.cbegin() + i * kc,
                    packB.cbegin() + j * kc,
                    C_itr.begin() + i * N + j);
            }
        }
    }

    template <typename T, size_t M, size_t N>
    static void micro_kernel_gemm(
        size_t k, size_t m, size_t n,
        typename VPANEL<T>::const_iterator &a,
        typename HPANEL<T>::const_iterator &b,
        typename tBLAS::MatrixXd<T, M, N>::iterator &C_itr)
    {
        for (l = 0; l < k; l++)
        {
            for (j = 0; j < n; j++)
            {
                for (i = 0; i < m; i++)
                {
                    (*C_itr)(i, j) += (*(a + l * m + i)) * (*(b + l * n + j));
                }
            }
        }
    }

}; // namespace tBLAS

#endif // BLAS_HPP