#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "matrix.hpp"
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

        /**
         * @brief Performs matmul based on kernel
         *
         * @note This wrapper allows us to add a automatic kernel selection at a later stage
         * without having to change the API
         *
         * @tparam T Type of the matrix
         * @tparam D Derived class of MatrixBase in CRTP
         * @tparam S Storage class of the matrix in the derived class D
         * @param A Left matrix
         * @param B Right matrix
         * @param C Output matrix
         * @param kernel Kernel to use
         */
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
            default:
                throw std::runtime_error("kernel not implemented!");
            }
        }
    }; // namespace BLAS
};     // namespace tBLAS
#endif