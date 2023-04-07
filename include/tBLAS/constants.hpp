#ifndef TBLAS_CONSTANTS_H
#define TBLAS_CONSTANTS_H
#include <cstddef>

namespace tBLAS
{
    const size_t KERNEL_MR = 8;
    const size_t KERNEL_NR = 4;
    const size_t KERNEL_NC = 64;
    const size_t KERNEL_MC = 7680;
    const size_t KERNEL_KC = 256;
    const size_t CACHE_LINE_SIZE = 64;

    const size_t VERTICAL_PANEL_ROWS = (KERNEL_MC / KERNEL_MR + 1) * KERNEL_MR;
    const size_t VERTICAL_PANEL_COLS = KERNEL_KC;
    const size_t VERTICAL_PANEL_SIZE = VERTICAL_PANEL_ROWS * VERTICAL_PANEL_COLS;

    const size_t HORIZONTAL_PANEL_ROWS = KERNEL_KC;
    const size_t HORIZONTAL_PANEL_COLS = (KERNEL_NC / KERNEL_NR + 1) * KERNEL_NR;
    const size_t HORIZONTAL_PANEL_SIZE = HORIZONTAL_PANEL_ROWS * HORIZONTAL_PANEL_COLS;

    const size_t Dynamic = 0;

}
#endif // TBLAS_CONSTANTS_H
