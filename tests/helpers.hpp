#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cstddef>
#include <random>

#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace tBLAS_test
{
    /**
     * @brief Generate a random matrix of size m x n.
     *
     * Random numbers are uniformly distributed between min and max params using Mersenne Twister engine.
     *
     * @tparam T Type of elements in the matrix
     * @param rows Number of rows
     * @param cols Number of columns
     * @param min Minimum value of the random numbers (default: -100)
     * @param max Maximum value of the random numbers (default: 100)
     * @return std::vector<std::vector<T>> Random matrix
     */
    template <typename T>
    static std::vector<std::vector<T>> gen_rand_matrix(const size_t rows, const size_t cols, const int min = -100, const int max = 100);

    /**
     * @brief Generate an identity matrix of size side x side.
     *
     * @tparam T Type of elements in the matrix
     * @param side Number of rows and columns
     * @return std::vector<std::vector<T>> Identity matrix of size side x side
     */
    template <typename T>
    static std::vector<std::vector<T>> identity(const size_t side);

    /**
     * @brief Apprx compare two 2D matrices
     *
     * This can be used to compare floating point types where and an exact equals might fail.
     *
     * @tparam T Type of elements in the matrix
     * @param exp Expected matrix
     * @param ip Input matrix
     */

    template <typename T, typename D, typename S, typename Container>
    static void compare_FP_2D(const Container &exp, const tBLAS::MatrixBase<T, D, S> &ip);
    // template <typename T>
    // static void compare_FP_2D(const std::vector<std::vector<T>> &exp, const std::vector<std::vector<T>> &ip);

    /**
     * @brief Compare two 2D matrices
     *
     * @param exp Expected matrix
     * @param ip Input matrix
     */
    template <typename T, typename D, typename S, typename Container>
    static void compare_INT_2D(const Container &exp, const tBLAS::MatrixBase<T, D, S> &ip);
    // static void compare_INT_2D(const std::vector<std::vector<int>> &exp, const std::vector<std::vector<int>> &ip);

    /* ----------------------------- Implementation ----------------------------- */
    template <typename T>
    static std::vector<std::vector<T>> gen_rand_matrix(const size_t rows, const size_t cols, const int min, const int max)
    {
        std::vector<std::vector<T>> R(rows, std::vector<T>(cols));

        std::random_device rd;  // Only used once to initialise (seed) engine
        std::mt19937 rng(rd()); // Random-number engine used (Mersenne-Twister in this case)
        std::uniform_int_distribution<int> uni(min, max);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; j++)
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

    template <typename T, typename D, typename S, typename Container>
    static void compare_FP_2D(const Container &exp, const tBLAS::MatrixBase<T, D, S> &ip)
    {
        for (int i = 0; i < exp.size(); ++i)
        {
            for (int j = 0; j < exp[0].size(); j++)
            {
                REQUIRE_THAT(ip(i, j), Catch::Matchers::WithinRel(exp[i][j]));
            }
        }
    }

    template <typename T, typename D, typename S, typename Container>
    static void compare_INT_2D(const Container &exp, const tBLAS::MatrixBase<T, D, S> &ip)
    {
        for (int i = 0; i < exp.size(); ++i)
        {
            for (int j = 0; j < exp[0].size(); j++)
            {
                REQUIRE(ip(i, j) == exp[i][j]);
            }
        }
    }

    // template <typename T>
    // static void compare_FP_2D(const std::vector<std::vector<T>> &exp, const std::vector<std::vector<T>> &ip)
    // {
    //     for (int i = 0; i < exp.size(); ++i)
    //     {
    //         for (int j = 0; j < exp[0].size(); j++)
    //         {
    //             REQUIRE_THAT(ip[i][j], Catch::Matchers::WithinRel(exp[i][j]));
    //         }
    //     }
    // }

    // static void compare_INT_2D(const std::vector<std::vector<int>> &exp, const std::vector<std::vector<int>> &ip)
    // {
    //     for (int i = 0; i < exp.size(); ++i)
    //     {
    //         for (int j = 0; j < exp[0].size(); j++)
    //         {
    //             REQUIRE(ip[i][j] == exp[i][j]);
    //         }
    //     }
    // }

}; // namespace tBLAS_test

#endif // HELPERS_HPP