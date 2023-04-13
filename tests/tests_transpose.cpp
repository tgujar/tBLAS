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
        size_t m = GENERATE(take(1, random(10, 50)));

        vector<vector<int>> A(gen_rand_matrix<int>(m, m));

        tBLAS::MatrixX<int> tA(A);

        SECTION("XL kernel")
        {
            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::xl));
        }
        SECTION("SM kernel")
        {
            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::sm));
        }
    }

    TEMPLATE_TEST_CASE("Transpose FP Square", "[transpose]", float, double)
    {
        using namespace std;
        size_t m = GENERATE(take(1, random(10, 50)));
        vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, m));

        tBLAS::MatrixX<TestType> tA(A);

        SECTION("XL kernel")
        {
            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::xl));
        }
        SECTION("SM kernel")
        {
            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::sm));
        }
    }

    TEST_CASE("Transpose INT Rectangle", "[transpose]")
    {
        using namespace std;
        size_t m = GENERATE(take(1, random(10, 50)));
        size_t n = GENERATE(take(1, random(10, 50)));

        vector<vector<int>> A(gen_rand_matrix<int>(m, n));

        tBLAS::MatrixX<int> tA(A);

        SECTION("XL kernel")
        {
            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::xl));
        }
        SECTION("SM kernel")
        {
            compare_INT_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::sm));
        }
    }

    TEMPLATE_TEST_CASE("Transpose FP Rectangle", "[transpose]", float, double)
    {
        using namespace std;
        size_t m = GENERATE(take(1, random(10, 50)));
        size_t n = GENERATE(take(1, random(10, 50)));

        vector<vector<TestType>> A(gen_rand_matrix<TestType>(m, n));
        tBLAS::MatrixX<TestType> tA(A);

        SECTION("XL kernel")
        {
            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::xl));
        }
        SECTION("SM kernel")
        {
            compare_FP_2D(tBLAS_test::Trivial::transpose(A), tBLAS::transpose(tA, tBLAS::BLAS::transpose_kernel::sm));
        }
    }
}; // namespace tBLAS_test