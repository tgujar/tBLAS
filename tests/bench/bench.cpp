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
const std::array<int, 2> large_powers_of_2 = {512, 1024};
const std::array<int, 5> primes = {31, 127, 257, 509, 1021};

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

    SECTION("Primes")
    {
        for (int i = 0; i < primes.size(); i++)
        {
            size_t m = primes[i];
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
                return tBLAS::BLAS::transpose(tA);
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
                return tBLAS::BLAS::transpose(tA);
            };
        }
    }

    SECTION("Primes")
    {
        for (int i = 0; i < primes.size(); i++)
        {
            size_t m = primes[i];
            vector<vector<double>> A(gen_rand_matrix<double>(m, m));
            tBLAS::MatrixX<double> tA(A);
            stringstream test_title;
            test_title << "[" << m << "x" << m << "]";
            BENCHMARK(test_title.str())
            {
                return tBLAS::BLAS::transpose(tA);
            };
        }
    }
}

TEST_CASE("Trivial Matmul", "[trivial][matmul][bench]")
{
    using namespace std;

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
                return tBLAS::BLAS::matmul(tA, tB);
            };
        }
    }
}