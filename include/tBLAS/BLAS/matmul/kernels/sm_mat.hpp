#ifndef SM_MATMUL_HPP
#define SM_MATMUL_HPP

#include <memory>

#include "../../../constants.hpp"
#include "../../../matrix.hpp"
#include "../../utils.hpp"
#include "../../../threading/pool.h"

namespace tBLAS
{
    namespace BLAS
    {

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