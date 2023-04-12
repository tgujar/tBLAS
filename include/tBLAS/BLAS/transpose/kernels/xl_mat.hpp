#ifndef XL_TRANSPOSE_HPP
#define XL_TRANSPOSE_HPP

#include <memory>

#include "../../../constants.hpp"
#include "../../../matrix.hpp"
#include "../../../utils.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        namespace detail
        {
            template <typename T>
            using xl_txp_VPANEL = Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>;
        }

        template <typename T, typename D, typename S>
        void micro_kernel_transpose(
            size_t k, size_t m,
            typename detail::xl_txp_VPANEL<T>::const_iterator a,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
            size_t C_stride);

        template <typename T, typename D, typename S>
        void macro_kernel_transpose(
            size_t mc, size_t kc,
            const detail::xl_txp_VPANEL<T> &packA,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
            size_t C_stride);

        template <typename T, typename D, typename S>
        void xl_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C);

        // template <typename T, size_t M, size_t K>
        // void simple_transpose(const Matrix<T, M, K> &A, Matrix<T, K, M> &C);

        template <typename T, typename D, typename S>
        void xl_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C)
        {
            auto &gtp = threading::GlobalThreadPool::get_instance();
            Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS> packA;
            for (size_t i_mc = 0; i_mc < A.rows(); i_mc += KERNEL_MC)
            {
                size_t mc = std::min(A.rows() - i_mc, KERNEL_MC);
                for (size_t i_kc = 0; i_kc < A.cols(); i_kc += KERNEL_KC)
                {
                    auto packA = std::make_shared<detail::xl_txp_VPANEL<T>>();
                    size_t kc = std::min(A.cols() - i_kc, KERNEL_KC);
                    for (size_t i_mr = 0; i_mr < mc; i_mr += KERNEL_MR)
                    {

                        pack_vertical<T, D, S>(A.cbegin() + i_kc + A.cols() * (i_mc + i_mr),
                                               packA->begin() + i_mr * kc,
                                               {std::min(mc - i_mr, KERNEL_MR), kc},
                                               A.cols());
                    }
                    gtp.enqueue([=, &C]() mutable
                                { macro_kernel_transpose<T, D, S>(
                                      mc, kc,
                                      *packA,
                                      C.begin() + i_mc + C.cols() * i_kc,
                                      C.cols()); });
                }
                gtp.sync();
            }
        }

        template <typename T, typename D, typename S>
        void macro_kernel_transpose(
            size_t mc, size_t kc,
            const detail::xl_txp_VPANEL<T> &packA,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
            size_t C_stride)
        {
            for (size_t i = 0; i < mc; i += KERNEL_MR)
            {
                micro_kernel_transpose<T, D, S>(
                    kc,
                    std::min(mc - i, KERNEL_MR),
                    packA.cbegin() + i * kc,
                    C_itr + i,
                    C_stride);
            }
        }

        template <typename T, typename D, typename S>
        void micro_kernel_transpose(
            size_t k, size_t m,
            typename detail::xl_txp_VPANEL<T>::const_iterator a,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
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

    };
}; // namespace tBLAS
#endif