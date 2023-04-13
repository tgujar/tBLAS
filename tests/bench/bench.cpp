#include <array>
#include <iostream>
#include <typeinfo>
#include <sstream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <tBLAS/blas.hpp>
#include <tBLAS/matrix.hpp>

#include "../trivial.hpp"
#include "../helpers.hpp"

const std::array<int, 9> powers_of_2 = {2, 4, 8, 16, 32, 64, 128, 256, 512};
const std::array<int, 2> large_powers_of_2 = {1024, 2048};

namespace tBLAS_test
{

    TEST_CASE("Trivial Transpose", "[trivial][transpose][bench]")
    {
        using namespace std;

        SECTION("Powers of 2")
        {
            for (int i = 0; i < powers_of_2.size(); i++)
            {
                size_t m = powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS_test::Trivial::transpose(A);
                };
            }
        }

        SECTION("Large Powers of 2")
        {
            for (int i = 0; i < large_powers_of_2.size(); i++)
            {
                size_t m = large_powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS_test::Trivial::transpose(A);
                };
            }
        }
    }

    TEST_CASE("tBLAS Transpose", "[tBLAS][transpose][bench]")
    {
        using namespace std;
        SECTION("Powers of 2")
        {
            for (int i = 0; i < powers_of_2.size(); i++)
            {
                size_t m = powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                tBLAS::MatrixX<double> tA(A);
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::sm);
                };
            }
        }

        SECTION("Large Powers of 2")
        {
            for (int i = 0; i < large_powers_of_2.size(); i++)
            {
                size_t m = large_powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                tBLAS::MatrixX<double> tA(A);
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::xl);
                };
            }
        }
    }

    TEST_CASE("Trivial Matmul", "[trivial][matmul][bench]")
    {
        using namespace std;

        SECTION("Powers of 2")
        {
            for (int i = 0; i < powers_of_2.size(); i++)
            {
                size_t m = powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                vector<vector<double>> B(gen_rand_matrix<double>(m, m));
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS_test::Trivial::matmul(A, B);
                };
            }
        }

        SECTION("Large Powers of 2")
        {
            for (int i = 0; i < large_powers_of_2.size(); i++)
            {
                size_t m = large_powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                vector<vector<double>> B(gen_rand_matrix<double>(m, m));
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS_test::Trivial::matmul(A, B);
                };
            }
        }
    }

    TEST_CASE("tBLAS Matmul", "[tBLAS][matmul][bench]")
    {
        using namespace std;

        SECTION("Powers of 2")
        {
            for (int i = 0; i < powers_of_2.size(); i++)
            {
                size_t m = powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                vector<vector<double>> B(gen_rand_matrix<double>(m, m));
                tBLAS::MatrixX<double> tA(A), tB(B);
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS::multiply(tA, tB, tBLAS::BLAS::matmul_kernel::sm);
                };
            }
        }

        SECTION("Large Powers of 2")
        {
            for (int i = 0; i < large_powers_of_2.size(); i++)
            {
                size_t m = large_powers_of_2[i];
                vector<vector<double>> A(gen_rand_matrix<double>(m, m));
                vector<vector<double>> B(gen_rand_matrix<double>(m, m));
                tBLAS::MatrixX<double> tA(A), tB(B);
                stringstream test_title;
                test_title << "[" << m << "x" << m << "]";
                BENCHMARK(test_title.str())
                {
                    return tBLAS::multiply(tA, tB, tBLAS::BLAS::matmul_kernel::xl);
                };
            }
        }
    }
}; // namespace tBLAS_test