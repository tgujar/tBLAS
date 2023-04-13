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
            xl,
        };

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
            default:
                throw std::runtime_error("kernel not implemented!");
            }
        }
    }; // namespace BLAS
};     // namespace tBLAS
#endif