#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>

#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

template <typename T, std::size_t M, std::size_t N>
std::array<std::array<T, N>, M> gen_rand_matrix()
{
    std::array<std::array<T, N>, M> R;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            R[i][j] = GENERATE(take(1, random(static_cast<T>(-100), static_cast<T>(100))));
        }
    }
    return R;
}

template <typename T>
std::vector<std::vector<T>> gen_rand_matrix(int m, int n)
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

template <typename T, std::size_t M, std::size_t N>
void compare_FP_2D(const std::array<std::array<T, N>, M> &exp, const std::array<std::array<T, N>, M> &ip)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            REQUIRE_THAT(ip[i][j], WithinRel(exp[i][j]));
        }
    }
}

template <typename T>
void compare_FP_2D(const std::vector<T> &exp, const std::vector<T> &ip)
{
    for (int i = 0; i < exp.size(); ++i)
    {
        for (int j = 0; j < exp[0].size(); j++)
        {
            REQUIRE_THAT(ip[i][j], Catch::Matchers::WithinRel(exp[i][j]));
        }
    }
}

#endif // UTILS_HPP