#ifndef XL_TRANSPOSE_HPP
#define XL_TRANSPOSE_HPP

#include <memory>

#include "../../../constants.hpp"
#include "../../../matrix.hpp"
#include "../../utils.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        namespace detail
        {
            template <typename T>
            using xl_txp_VPANEL = Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>;
        }

        /**
         * @brief Helper function for macro_kernel_transpose. Computes matrix transpose for a micro kernel of size KERNEL_MR x KERNEL_KR.
         *
         * @tparam T The type of the matrix elements.
         * @tparam D Derived class of MatrixBase in CRTP
         * @tparam S Storage class of the matrix in the derived class D
         * @param k The number of colums in the current micro kernel <= KERNEL_KR.
         * @param m The number of rows in the current micor kernel <= KERNEL_MR.
         * @param a The iterator to the left matrix.
         * @param C_itr The iterator to the result matrix.
         * @param C_stride Number of columns of the result matrix.
         */
        template <typename T, typename D, typename S>
        void micro_kernel_transpose(
            size_t k, size_t m,
            typename detail::xl_txp_VPANEL<T>::const_iterator a,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
            size_t C_stride);

        /**
         * @brief Helper function for xl_txp. Computes matrix transpose for a macro kernel of size KERNEL_MC x KERNEL_KC.
         *
         * @tparam T The type of the matrix elements.
         * @tparam D Derived class of MatrixBase in CRTP
         * @tparam S Storage class of the matrix in the derived class D
         * @param mc The number of rows in the current macro kernel <= KERNEL_MC.
         * @param kc The number of columns in the current macro kernel <= KERNEL_KC.
         * @param packA The packed left matrix.
         * @param C_itr The iterator to the result matrix.
         * @param C_stride Number of columns of the result matrix.
         */
        template <typename T, typename D, typename S>
        void macro_kernel_transpose(
            size_t mc, size_t kc,
            const detail::xl_txp_VPANEL<T> &packA,
            typename tBLAS::MatrixBase<T, D, S>::iterator C_itr,
            size_t C_stride);

        /**
         * @brief Cache optimized and multithreaded matrix transpose. To be used for large matrices.
         *
         * @tparam T The type of the matrix elements.
         * @tparam D Derived class of MatrixBase in CRTP.
         * @tparam S Storage class of the matrix in the derived class D
         * @param A The left matrix.
         * @param C The result matrix.
         */
        template <typename T, typename D, typename S>
        void xl_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C);

        /* ----------------------------- Implementation ----------------------------- */
        template <typename T, typename D, typename S>
        void xl_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C)
        {
            auto &gtp = threading::GlobalThreadPool::get_instance();
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