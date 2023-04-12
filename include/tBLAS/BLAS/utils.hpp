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
    /**
     * @brief Helper function to pack a matrix into a vertical panel of size VERTICAL_PANEL_ROWS x VERTICAL_PANEL_COLS
     *
     * @tparam T Type of the matrix
     * @tparam D Derived class of MatrixBase in CRTP
     * @tparam S Storage class of the matrix in the derived class D
     * @param matrix_itr Iterator to the input matrix
     * @param panel_itr Iterator to the output panel
     * @param panel_dims Dimensions of the panel in the input matrix
     * @param stride Number of colums in the input matrix
     */
    template <typename T, typename D, typename S>
    inline void pack_vertical(typename tBLAS::MatrixBase<T, D, S>::const_iterator matrix_itr,
                              typename tBLAS::Matrix<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>::iterator panel_itr,
                              const std::pair<size_t, size_t> &panel_dims,
                              size_t stride)
    {

        for (size_t i_k = 0; i_k < panel_dims.second; ++i_k)
        {
            for (size_t i_m = 0; i_m < panel_dims.first; ++i_m)
            {
                *panel_itr = *(matrix_itr + i_m * stride + i_k);
                panel_itr++;
            }
        }
    }

    /**
     * @brief Helper function to pack a matrix into a horizontal panel of size HORIZONTAL_PANEL_ROWS x HORIZONTAL_PANEL_COLS
     *
     * @tparam T Type of the matrix
     * @tparam D Derived class of MatrixBase in CRTP
     * @tparam S Storage class of the matrix in the derived class D
     * @param matrix_itr Iterator to the input matrix
     * @param panel_itr Iterator to the output panel
     * @param panel_dims Dimensions of the panel in the input matrix
     * @param stride Number of colums in the input matrix
     */
    template <typename T, typename D, typename S>
    inline void pack_horizontal(typename tBLAS::MatrixBase<T, D, S>::const_iterator matrix_itr,
                                typename tBLAS::Matrix<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>::iterator panel_itr,
                                const std::pair<size_t, size_t> &panel_dims,
                                size_t stride)
    {
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
