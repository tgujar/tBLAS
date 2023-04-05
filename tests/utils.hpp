#ifndef UTILS_HPP
#define UTILS_HPP

#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

template <typename T>
std::vector<std::vector<T>> genRandMatrix(int m, int n)
{
    std::vector<std::vector<T>> R(m, std::vector<T>(n));
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; j++)
        {
            R[i][j] = GENERATE(take(1, random(static_cast<T>(-100), static_cast<T>(100))));
        }
    }
    return R;
}

#endif // UTILS_HPP