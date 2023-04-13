#ifndef SM_MATMUL_HPP
#define SM_MATMUL_HPP

#include "constants.hpp"
#include "matrix.hpp"
#include "BLAS/utils.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        /**
         * @brief Performs matrix multiplication of two matrices.
         *
         * To be used for small matrices which might fit into L1 cache.
         *
         * @tparam T Type of the matrix
         * @tparam D Derived class of MatrixBase in CRTP.
         * @tparam S Storage class of the matrix in the derived class D.
         * @param A Left matrix
         * @param B Right matrix
         * @param C Result matrix
         */
        template <typename T, typename D, typename S>
        void sm_gemm(const MatrixBase<T, D, S> &A, const MatrixBase<T, D, S> &B, MatrixBase<T, D, S> &C);

        /* ----------------------------- Implementation ----------------------------- */

        template <typename T, typename D, typename S>
        void sm_gemm(const MatrixBase<T, D, S> &A, const MatrixBase<T, D, S> &B, MatrixBase<T, D, S> &C)
        {
            for (size_t k = 0; k < A.cols(); k++)
            {
                for (size_t m = 0; m < A.rows(); m++)
                {
                    for (size_t n = 0; n < B.cols(); n++)
                    {
                        C(m, n) += A(m, k) * B(k, n);
                    }
                }
            }
        }

    }; // namespace BLAS
};     // namespace tBLAS
#endif