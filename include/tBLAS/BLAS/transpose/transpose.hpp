#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include "matrix.hpp"
#include "./kernels/xl_mat.hpp"
#include "./kernels/sm_mat.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        enum class transpose_kernel
        {
            sm,
            xl
        };

        /**
         * @brief Calls appropriate kernel to perform out of place transpose of a matrix
         *
         * @tparam T Type of the matrix
         * @tparam D Derived class of MatrixBase in CRTP.
         * @tparam S Storage class of the matrix in the derived class D.
         * @param A The matrix to be transposed.
         * @param C The result matrix.
         * @param kernel The kernel to be used for the operation.
         */
        template <typename T, typename D, typename S>
        void matrix_transpose(
            const MatrixBase<T, D, S> &A,
            MatrixBase<T, D, S> &C,
            transpose_kernel kernel);

        /* ----------------------------- Implementation ----------------------------- */

        template <typename T, typename D, typename S>
        void matrix_transpose(
            const MatrixBase<T, D, S> &A,
            MatrixBase<T, D, S> &C,
            transpose_kernel kernel)
        {
            switch (kernel)
            {
            case transpose_kernel::xl:
                xl_txp(A, C);
                break;
            case transpose_kernel::sm:
                sm_txp(A, C);
                break;
            default:
                throw std::runtime_error("kernel not implemented!");
            }
        }
    }; // namespace BLAS
};     // namespace tBLAS
#endif