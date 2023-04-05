#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/blas.hpp>
#include "trivial.hpp"
#include "utils.hpp"

TEST_CASE("Matmul trivial", "[matmul]")
{

    SECTION("Square matrices")
    {
        using namespace std;
        vector<vector<int>> A = {{1, 2}, {3, 4}};
        vector<vector<int>> B = {{1, 2}, {3, 4}};
        REQUIRE(tBLAS_test::Trivial<int>::matmul(A, B) == vector<vector<int>>{{7, 10}, {15, 22}});
    }
    SECTION("Rectangular matrices")
    {
        using namespace std;
        vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
        vector<vector<int>> B = {{1, 2}, {3, 4}, {5, 6}};
        REQUIRE(tBLAS_test::Trivial<int>::matmul(A, B) == vector<vector<int>>{{22, 28}, {49, 64}});
    }
}

TEMPLATE_TEST_CASE("Matmul Square Random", "[matmul]", int, float, double)
{
    using namespace std;
    for (int i = 0; i < 10; i++)
    {
        int m = GENERATE(take(1, random(10, 50)));

        vector<vector<TestType>> A = genRandMatrix<TestType>(m, m);
        vector<vector<TestType>> B = genRandMatrix<TestType>(m, m);

        REQUIRE(tBLAS_test::Trivial<TestType>::matmul(A, B) == tBLAS::BLAS<TestType>::matmul(A, B));
    }
}

TEMPLATE_TEST_CASE("Matmul Rectangle Random", "[matmul]", int, float, double)
{
    using namespace std;
    for (int i = 0; i < 10; i++)
    {
        int m = GENERATE(take(1, random(10, 50)));
        int n = GENERATE(take(1, random(10, 50)));
        int p = GENERATE(take(1, random(10, 50)));

        vector<vector<TestType>> A = genRandMatrix<TestType>(m, n);
        vector<vector<TestType>> B = genRandMatrix<TestType>(n, p);

        REQUIRE(tBLAS_test::Trivial<TestType>::matmul(A, B) == tBLAS::BLAS<TestType>::matmul(A, B));
    }
}
