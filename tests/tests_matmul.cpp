#include <array>
#include <iostream>
#include <sstream>
#include <typeinfo>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/blas.hpp>
#include <tBLAS/matrix.hpp>

#include "trivial.hpp"
#include "helpers.hpp"

namespace tBLAS_test
{
    TEST_CASE("Matmul trivial", "[matmul]")
    {
        SECTION("Square matrices")
        {
            using namespace std;
            vector<vector<int>> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            vector<vector<int>> B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

            REQUIRE(tBLAS_test::Trivial::matmul(A, B) == vector<vector<int>>{{{30, 36, 42}, {66, 81, 96}, {102, 126, 150}}});
        }
        SECTION("Rectangular matrices")
        {
            using namespace std;
            vector<vector<int>> A = {{{1, 2, 3}, {4, 5, 6}}};
            vector<vector<int>> B = {{{1, 2}, {3, 4}, {5, 6}}};
            REQUIRE(tBLAS_test::Trivial::matmul(A, B) == vector<vector<int>>{{{22, 28}, {49, 64}}});
        }
        SECTION("Identity")
        {
            using namespace std;
            for (int i = 0; i < 10; i++)
            {
                size_t m = GENERATE(take(1, random(10, 50)));
                size_t n = GENERATE(take(1, random(10, 50)));

                vector<vector<int>> A(gen_rand_matrix<int>(m, n));
                vector<vector<int>> B(identity<int>(n));

                compare_INT_2D(tBLAS_test::Trivial::matmul(A, B), A);
            }
        }
    }

    TEST_CASE("Matmul INT Square", "[matmul]")
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));

            vector<vector<int>> A(gen_rand_matrix<int>(m, m));
            vector<vector<int>> B(gen_rand_matrix<int>(m, m));

            tBLAS::MatrixX<int> tA(A);
            tBLAS::MatrixX<int> tB(B);

            compare_INT_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::multiply(tA, tB).to_vector());
        }
    }

    TEMPLATE_TEST_CASE("Matmul FP Square", "[matmul]", float, double)
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));

            vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, m));
            vector<vector<TestType>> B(gen_rand_matrix<TestType>(m, m));

            tBLAS::MatrixX<TestType> tA(A);
            tBLAS::MatrixX<TestType> tB(B);

            compare_FP_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::multiply(tA, tB).to_vector());
        }
    }

    TEST_CASE("Matmul INT Rectangle", "[matmul]")
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));
            size_t n = GENERATE(take(1, random(10, 50)));
            size_t p = GENERATE(take(1, random(10, 50)));

            vector<vector<int>> A(gen_rand_matrix<int>(m, n));
            vector<vector<int>> B(gen_rand_matrix<int>(n, p));

            tBLAS::MatrixX<int> tA(A);
            tBLAS::MatrixX<int> tB(B);

            compare_INT_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::multiply(tA, tB).to_vector());
        }
    }

    TEMPLATE_TEST_CASE("Matmul FP Rectangle", "[matmul]", float, double)
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));
            size_t n = GENERATE(take(1, random(10, 50)));
            size_t p = GENERATE(take(1, random(10, 50)));

            vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, n));
            vector<vector<TestType>> B(gen_rand_matrix<TestType>(n, p));

            tBLAS::MatrixX<TestType> tA(A);
            tBLAS::MatrixX<TestType> tB(B);

            compare_FP_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::multiply(tA, tB).to_vector());
        }
    }
}; // namespace tBLAS_test
