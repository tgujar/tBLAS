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

    // template <typename T, std::size_t M, std::size_t K>
    // tBLAS::MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>
    // pack_vertical(const tBLAS::MatrixXd<T, M, K> &A,
    //               const std::pair<size_t, size_t> &panel_dims,
    //               const std::pair<size_t, size_t> &panel_start)
    // {
    //     assert(panel_dims.first <= KERNEL_MR);
    //     assert(panel_dims.second <= KERNEL_KC);
    //     assert(panel_start.first <= M);
    //     assert(panel_start.second <= K);

    //     tBLAS::MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS> panel;
    //     auto panel_itr = panel.begin();
    //     auto matrix_itr = A.cbegin() + (panel_start.first * K + panel_start.second);

    //     for (size_t i_k = 0; i_k != panel_dims.second; ++i_k)
    //     {
    //         for (size_t i_m = 0; i_m != panel_dims.first; ++i_m)
    //         {
    //             *panel_itr = *(matrix_itr + i_m * K + i_k);
    //             panel_itr++;
    //         }
    //     }
    //     return panel;
    // }

    // template <typename T, std::size_t K, std::size_t N>
    // tBLAS::MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>
    // pack_horizontal(const tBLAS::MatrixXd<T, K, N> &B,
    //                 const std::pair<size_t, size_t> &panel_dims,
    //                 const std::pair<size_t, size_t> &panel_start)
    // {
    //     assert(panel_dims.first <= KERNEL_KC);
    //     assert(panel_dims.second <= KERNEL_NR);
    //     assert(panel_start.first <= K);
    //     assert(panel_start.second <= N);

    //     tBLAS::MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS> panel;

    //     auto panel_itr = panel.begin();
    //     auto matrix_itr = B.cbegin() + (panel_start.first * N + panel_start.second);

    //     for (size_t i_k = 0; i_k != panel_dims.first; ++i_k)
    //     {
    //         for (size_t i_n = 0; i_n != panel_dims.second; ++i_n)
    //         {
    //             *panel_itr = *(matrix_itr + i_n + i_k * N);
    //             panel_itr++;
    //         }
    //     }
    //     return panel;
    // }

    template <typename T, std::size_t M, std::size_t K>
    void pack_vertical(typename tBLAS::MatrixXd<T, M, K>::const_iterator matrix_itr,
                       typename tBLAS::MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS>::iterator panel_itr,
                       const std::pair<size_t, size_t> &panel_dims)
    {
        assert(panel_dims.first <= KERNEL_MR);
        assert(panel_dims.second <= KERNEL_KC);

        tBLAS::MatrixXd<T, VERTICAL_PANEL_ROWS, VERTICAL_PANEL_COLS> panel;

        for (size_t i_k = 0; i_k < panel_dims.second; ++i_k)
        {
            for (size_t i_m = 0; i_m < panel_dims.first; ++i_m)
            {
                *panel_itr = *(matrix_itr + i_m * K + i_k);
                panel_itr++;
            }
        }
    }

    template <typename T, std::size_t K, std::size_t N>
    void pack_horizontal(typename tBLAS::MatrixXd<T, K, N>::const_iterator matrix_itr,
                         typename tBLAS::MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS>::iterator panel_itr,
                         const std::pair<size_t, size_t> &panel_dims)
    {
        assert(panel_dims.first <= KERNEL_KC);
        assert(panel_dims.second <= KERNEL_NR);

        tBLAS::MatrixXd<T, HORIZONTAL_PANEL_ROWS, HORIZONTAL_PANEL_COLS> panel;

        for (size_t i_k = 0; i_k < panel_dims.first; ++i_k)
        {
            for (size_t i_n = 0; i_n < panel_dims.second; ++i_n)
            {
                *panel_itr = *(matrix_itr + i_k * N + i_n);
                panel_itr++;
            }
        }
    }

}

#endif // TBLAS_UTILS_H
