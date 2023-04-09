#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <memory>

#include "../constants.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"
#include "../threading/pool.h"

namespace tBLAS
{
    namespace BLAS
    {

        template <typename T>
        using VPANEL = Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>;
        template <typename T>
        using HPANEL = Matrix<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>;

        /**
         * @brief Cache optimized and multithreaded matrix multiplication. To be used for large matrices.
         *
         * @tparam T The type of the matrix elements.
         * @tparam M The number of rows of the left matrix.
         * @tparam N The number of columns of the right matrix.
         * @tparam K The number of columns of the left matrix and the number of rows of the right matrix.
         * @param A The left matrix.
         * @param B The right matrix.
         * @param C The result matrix.
         */
        template <typename T, size_t M, size_t N, size_t K>
        void matrix_gemm(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B, Matrix<T, M, N> &C);

        /**
         * @brief Helper function for matrix_gemm. Computes matrix multiply for a macro kernel of size KERNEL_MC x KERNEL_NC.
         *
         * @tparam T The type of the matrix elements.
         * @tparam M The number of rows of the left matrix.
         * @tparam N The number of columns of the right matrix.
         * @param mc The number of rows of the left matrix.
         * @param nc The number of columns of the right matrix.
         * @param kc The number of columns of the left matrix and the number of rows of the right matrix.
         * @param packA The packed left matrix.
         * @param packB The packed right matrix.
         * @param C_itr The iterator to the result matrix.
         * @param C_stride Number of columns of the result matrix.
         */
        template <typename T, size_t M, size_t N>
        void macro_kernel_gemm(
            size_t mc, size_t nc, size_t kc,
            const VPANEL<T> &packA,
            const HPANEL<T> &packB,
            typename tBLAS::Matrix<T, M, N>::iterator C_itr,
            size_t C_stride);

        /**
         * @brief Helper function for macro_kernel_gemm. Computes matrix multiply for a micro kernel of size KERNEL_MR x KERNEL_NR.
         *
         * @tparam T The type of the matrix elements.
         * @tparam M The number of rows of the left matrix.
         * @tparam N The number of columns of the right matrix.
         * @param k The number of columns of the left matrix and the number of rows of the right matrix.
         * @param m The number of rows of the left matrix.
         * @param n The number of columns of the right matrix.
         * @param a The iterator to the left matrix.
         * @param b The iterator to the right matrix.
         * @param C_itr The iterator to the result matrix.
         * @param C_stride Number of columns of the result matrix.
         */
        template <typename T, size_t M, size_t N>
        void micro_kernel_gemm(
            size_t k, size_t m, size_t n,
            typename VPANEL<T>::const_iterator a,
            typename HPANEL<T>::const_iterator b,
            typename tBLAS::Matrix<T, M, N>::iterator C_itr,
            size_t C_stride);

        /* ----------------------------- Implementation ----------------------------- */

        template <typename T, size_t M, size_t N, size_t K>
        void matrix_gemm(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B, Matrix<T, M, N> &C)
        {
            auto &gtp = threading::GlobalThreadPool::get_instance();
            VPANEL<T> packA;
            for (size_t i_mc = 0; i_mc < A.rows(); i_mc += KERNEL_MC)
            {
                size_t mc = std::min(A.rows() - i_mc, KERNEL_MC);
                for (size_t i_kc = 0; i_kc < A.cols(); i_kc += KERNEL_KC)
                {
                    size_t kc = std::min(A.cols() - i_kc, KERNEL_KC);
                    for (size_t i_mr = 0; i_mr < mc; i_mr += KERNEL_MR)
                    {
                        pack_vertical<T, M, K>(A.cbegin() + i_kc + A.cols() * (i_mc + i_mr),
                                               packA.begin() + i_mr * kc,
                                               {std::min(mc - i_mr, KERNEL_MR), kc},
                                               A.cols());
                    }

                    for (size_t i_nc = 0; i_nc < B.cols(); i_nc += KERNEL_NC)
                    {
                        auto packB = std::make_shared<HPANEL<T>>();
                        size_t nc = std::min(B.cols() - i_nc, KERNEL_NC);
                        for (size_t i_nr = 0; i_nr < nc; i_nr += KERNEL_NR)
                        {
                            pack_horizontal<T, K, N>(B.cbegin() + i_kc * B.cols() + i_nc + i_nr,
                                                     packB->begin() + i_nr * kc,
                                                     {kc, std::min(nc - i_nr, KERNEL_NR)},
                                                     B.cols());
                        }
                        gtp.enqueue([=, &packA, &C]() mutable
                                    { macro_kernel_gemm<T, M, N>(mc,
                                                                 nc,
                                                                 kc,
                                                                 packA,
                                                                 *(packB),
                                                                 C.begin() + i_mc * C.cols() + i_nc,
                                                                 C.cols()); });
                    }
                    gtp.sync();
                }
            }
        }

        template <typename T, size_t M, size_t N>
        inline void macro_kernel_gemm(
            size_t mc, size_t nc, size_t kc,
            const VPANEL<T> &packA,
            const HPANEL<T> &packB,
            typename tBLAS::Matrix<T, M, N>::iterator C_itr,
            size_t C_stride)
        {
            for (size_t i = 0; i < mc; i += KERNEL_MR)
            {
                for (size_t j = 0; j < nc; j += KERNEL_NR)
                {
                    micro_kernel_gemm<T, M, N>(
                        kc,
                        std::min(mc - i, KERNEL_MR),
                        std::min(nc - j, KERNEL_NR),
                        packA.cbegin() + i * kc,
                        packB.cbegin() + j * kc,
                        C_itr + i * C_stride + j,
                        C_stride);
                }
            }
        }

        template <typename T, size_t M, size_t N>
        inline void micro_kernel_gemm(
            size_t k, size_t m, size_t n,
            typename VPANEL<T>::const_iterator a,
            typename HPANEL<T>::const_iterator b,
            typename tBLAS::Matrix<T, M, N>::iterator C_itr,
            size_t C_stride)
        {
            for (size_t l = 0; l < k; l++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    for (size_t i = 0; i < m; i++)
                    {
                        *(C_itr + i * C_stride + j) += (*(a + l * m + i)) * (*(b + l * n + j));
                    }
                }
            }
        }
    }; // namespace BLAS
};     // namespace tBLAS
#endif