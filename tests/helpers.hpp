#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cstddef>
#include <random>

#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

template <typename T>
static std::vector<std::vector<T>> gen_rand_matrix(const size_t m, const size_t n)
{
    std::vector<std::vector<T>> R(m, std::vector<T>(n));

    std::random_device rd;  // Only used once to initialise (seed) engine
    std::mt19937 rng(rd()); // Random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(-100, 100);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; j++)
        {
            R[i][j] = uni(rng);
        }
    }
    return R;
}

template <typename T>
static std::vector<std::vector<T>> identity(const size_t m)
{
    std::vector<std::vector<T>> R(m, std::vector<T>(m));
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; j++)
        {
            if (i == j)
            {
                R[i][j] = 1;
            }
            else
            {
                R[i][j] = 0;
            }
        }
    }
    return R;
}

template <typename T>
static void compare_FP_2D(const std::vector<std::vector<T>> &exp, const std::vector<std::vector<T>> &ip)
{
    for (int i = 0; i < exp.size(); ++i)
    {
        for (int j = 0; j < exp[0].size(); j++)
        {
            REQUIRE_THAT(ip[i][j], Catch::Matchers::WithinRel(exp[i][j]));
        }
    }
}

static void compare_INT_2D(const std::vector<std::vector<int>> &exp, const std::vector<std::vector<int>> &ip)
{
    for (int i = 0; i < exp.size(); ++i)
    {
        for (int j = 0; j < exp[0].size(); j++)
        {
            REQUIRE(ip[i][j] == exp[i][j]);
        }
    }
}

#endif // HELPERS_HPP