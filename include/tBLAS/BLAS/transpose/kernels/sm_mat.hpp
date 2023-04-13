#ifndef SM_TRANSPOSE_HPP
#define SM_TRANSPOSE_HPP

#include "constants.hpp"
#include "matrix.hpp"
#include "BLAS/utils.hpp"

namespace tBLAS
{
    namespace BLAS
    {
        /**
         * @brief Performs out of place transpose of a matrix
         *
         * To be used for small matrices which might fit into L1 cache.
         *
         * @tparam T Type of the matrix
         * @tparam D Derived class of MatrixBase in CRTP.
         * @tparam S Storage class of the matrix in the derived class D.
         * @param A The matrix to be transposed.
         * @param C The result matrix.
         */
        template <typename T, typename D, typename S>
        void sm_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C);

        /* ----------------------------- Implementation ----------------------------- */

        template <typename T, typename D, typename S>
        void sm_txp(const MatrixBase<T, D, S> &A, MatrixBase<T, D, S> &C)
        {
            for (size_t i = 0; i < A.rows(); ++i)
            {
                for (size_t j = 0; j < A.cols(); ++j)
                {
                    C(j, i) = A(i, j);
                }
            }
        }

    }; // namespace BLAS
};     // namespace tBLAS
#endif