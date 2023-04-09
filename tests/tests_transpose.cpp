#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <vector>

#include <tBLAS/blas.hpp>
#include <sstream>
#include "trivial.hpp"
#include "helpers.hpp"

namespace tBLAS_test
{
    TEST_CASE("Transpose trivial", "[transpose]")
    {

        SECTION("Square matrices")
        {
            using namespace std;
            vector<vector<int>> A = {{1, 2}, {3, 4}};
            REQUIRE(tBLAS_test::Trivial::transpose(A) == vector<vector<int>>{{1, 3}, {2, 4}});
        }
        SECTION("Rectangular matrices")
        {
            using namespace std;
            vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
            REQUIRE(tBLAS_test::Trivial::transpose(A) == vector<vector<int>>{{1, 4}, {2, 5}, {3, 6}});
        }
    }

    TEST_CASE("Transpose INT Square", "[transpose]")
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));

            vector<vector<int>> A(gen_rand_matrix<int>(m, m));

            tBLAS::MatrixX<int> tA(A);

            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::BLAS::transpose(tA).to_vector());
        }
    }

    TEMPLATE_TEST_CASE("Transpose FP Square", "[matmul]", float, double)
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));

            vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, m));

            tBLAS::MatrixX<TestType> tA(A);

            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::BLAS::transpose(tA).to_vector());
        }
    }

    TEST_CASE("Transpose INT Rectangle", "[transpose]")
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));
            size_t n = GENERATE(take(1, random(10, 50)));

            vector<vector<int>> A(gen_rand_matrix<int>(m, n));

            tBLAS::MatrixX<int> tA(A);

            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::BLAS::transpose(tA).to_vector());
        }
    }

    TEMPLATE_TEST_CASE("Transpose FP Rectangle", "[matmul]", float, double)
    {
        using namespace std;
        for (int i = 0; i < 10; i++)
        {
            size_t m = GENERATE(take(1, random(10, 50)));
            size_t n = GENERATE(take(1, random(10, 50)));

            vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, n));

            tBLAS::MatrixX<TestType> tA(A);

            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::BLAS::transpose(tA).to_vector());
        }
    }
}; // namespace tBLAS_test