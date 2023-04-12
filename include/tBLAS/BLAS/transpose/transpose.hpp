#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <memory>

#include "../constants.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"

namespace tBLAS
{
    namespace BLAS
    {

        template <typename T>
        using VPANEL = Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>;
        template <typename T>
        using HPANEL = Matrix<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>;

        template <typename T, size_t M, size_t K>
        void micro_kernel_transpose(
            size_t k, size_t m,
            typename VPANEL<T>::const_iterator a,
            typename tBLAS::Matrix<T, K, M>::iterator C_itr,
            size_t C_stride);

        template <typename T, size_t M, size_t K>
        void macro_kernel_transpose(
            size_t mc, size_t kc,
            const VPANEL<T> &packA,
            typename tBLAS::Matrix<T, K, M>::iterator C_itr,
            size_t C_stride);

        template <typename T, size_t M, size_t K>
        void matrix_transpose(const Matrix<T, M, K> &A, Matrix<T, K, M> &C);

        template <typename T, size_t M, size_t K>
        void simple_transpose(const Matrix<T, M, K> &A, Matrix<T, K, M> &C);

        template <typename T, size_t M, size_t K>
        void matrix_transpose(const Matrix<T, M, K> &A, Matrix<T, K, M> &C)
        {
            auto &gtp = threading::GlobalThreadPool::get_instance();
            Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS> packA;
            for (size_t i_mc = 0; i_mc < A.rows(); i_mc += KERNEL_MC)
            {
                size_t mc = std::min(A.rows() - i_mc, KERNEL_MC);
                for (size_t i_kc = 0; i_kc < A.cols(); i_kc += KERNEL_KC)
                {
                    auto packA = std::make_shared<VPANEL<T>>();
                    size_t kc = std::min(A.cols() - i_kc, KERNEL_KC);
                    for (size_t i_mr = 0; i_mr < mc; i_mr += KERNEL_MR)
                    {

                        pack_vertical<T, M, K>(A.cbegin() + i_kc + A.cols() * (i_mc + i_mr),
                                               packA->begin() + i_mr * kc,
                                               {std::min(mc - i_mr, KERNEL_MR), kc},
                                               A.cols());
                    }
                    gtp.enqueue([=, &C]() mutable
                                { macro_kernel_transpose<T, M, K>(
                                      mc, kc,
                                      *packA,
                                      C.begin() + i_mc + C.cols() * i_kc,
                                      C.cols()); });
                }
                gtp.sync();
            }
        }

        template <typename T, size_t M, size_t K>
        void macro_kernel_transpose(
            size_t mc, size_t kc,
            const VPANEL<T> &packA,
            typename tBLAS::Matrix<T, K, M>::iterator C_itr,
            size_t C_stride)
        {
            for (size_t i = 0; i < mc; i += KERNEL_MR)
            {
                micro_kernel_transpose<T, M, K>(
                    kc,
                    std::min(mc - i, KERNEL_MR),
                    packA.cbegin() + i * kc,
                    C_itr + i,
                    C_stride);
            }
        }

        template <typename T, size_t M, size_t K>
        void micro_kernel_transpose(
            size_t k, size_t m,
            typename VPANEL<T>::const_iterator a,
            typename tBLAS::Matrix<T, K, M>::iterator C_itr,
            size_t C_stride)
        {
            for (size_t l = 0; l < k; l++)
            {
                for (size_t i = 0; i < m; i++)
                {
                    *(C_itr + l * C_stride + i) = (*(a + l * m + i));
                }
            }
        }

        // template <typename T, size_t M, size_t K>
        // void simple_transpose(const Matrix<T, M, K> &A, Matrix<T, K, M> &C){

        // }

    };
}; // namespace tBLAS
#endif