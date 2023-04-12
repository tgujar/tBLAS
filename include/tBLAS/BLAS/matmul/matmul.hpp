#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <memory>

#include "../../matrix.hpp"
#include "./kernels/xl_mat.hpp"
#include "./kernels/sm_mat.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        enum class matmul_kernel
        {
            xl,
            sm
        };

        template <typename T, typename D, typename S>
        void matmul(
            const tBLAS::MatrixBase<T, D, S> &A,
            const tBLAS::MatrixBase<T, D, S> &B,
            tBLAS::MatrixBase<T, D, S> &C,
            matmul_kernel kernel);

        /* ----------------------------- Implementation ----------------------------- */

        template <typename T, typename D, typename S>
        void matmul(
            const tBLAS::MatrixBase<T, D, S> &A,
            const tBLAS::MatrixBase<T, D, S> &B,
            tBLAS::MatrixBase<T, D, S> &C,
            matmul_kernel kernel)
        {
            switch (kernel)
            {
            case matmul_kernel::xl:
                xl_gemm(A, B, C);
                break;
            case matmul_kernel::sm:
                sm_gemm(A, B, C);
                break;
            }
        }
    }; // namespace BLAS
};     // namespace tBLAS
#endif