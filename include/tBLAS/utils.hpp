#ifndef TBLAS_UTILS_H
#define TBLAS_UTILS_H

#include "tBLAS/constants.hpp"
#include "tBLAS/matrix.hpp"
#include <array>
#include <vector>
#include <utility>
#include <cstddef>
#include <cassert>

namespace tBLAS
{

    template <typename T, typename D, typename S>
    static inline void pack_vertical(typename tBLAS::MatrixBase<T, D, S>::const_iterator matrix_itr,
                                     typename tBLAS::Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>::iterator panel_itr,
                                     const std::pair<size_t, size_t> &panel_dims,
                                     size_t stride)
    {
        // assert(panel_dims.first <= KERNEL_MR);
        // assert(panel_dims.second <= KERNEL_KC);

        for (size_t i_k = 0; i_k < panel_dims.second; ++i_k)
        {
            for (size_t i_m = 0; i_m < panel_dims.first; ++i_m)
            {
                *panel_itr = *(matrix_itr + i_m * stride + i_k);
                panel_itr++;
            }
        }
    }

    template <typename T, typename D, typename S>
    static inline void pack_horizontal(typename tBLAS::MatrixBase<T, D, S>::const_iterator matrix_itr,
                                       typename tBLAS::Matrix<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>::iterator panel_itr,
                                       const std::pair<size_t, size_t> &panel_dims,
                                       size_t stride)
    {
        // assert(panel_dims.first <= KERNEL_KC);
        // assert(panel_dims.second <= KERNEL_NR);

        for (size_t i_k = 0; i_k < panel_dims.first; ++i_k)
        {
            for (size_t i_n = 0; i_n < panel_dims.second; ++i_n)
            {
                *panel_itr = *(matrix_itr + i_k * stride + i_n);
                panel_itr++;
            }
        }
    }

}

#endif // TBLAS_UTILS_H
