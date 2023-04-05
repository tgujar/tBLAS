#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/blas.hpp>
#include "trivial.hpp"
#include "utils.hpp"

TEST_CASE("Transpose trivial", "[transpose]")
{

    SECTION("Square matrices")
    {
        using namespace std;
        vector<vector<int>> A = {{1, 2}, {3, 4}};
        REQUIRE(tBLAS_test::Trivial<int>::transpose(A) == vector<vector<int>>{{1, 3}, {2, 4}});
    }
    SECTION("Rectangular matrices")
    {
        using namespace std;
        vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
        REQUIRE(tBLAS_test::Trivial<int>::transpose(A) == vector<vector<int>>{{1, 4}, {2, 5}, {3, 6}});
    }
}

TEMPLATE_TEST_CASE("Transpose Square Random", "[transpose]", int, float, double)
{

    using namespace std;
    for (int i = 0; i < 10; i++)
    {
        int m = GENERATE(take(1, random(50, 100)));

        vector<vector<TestType>> A = genRandMatrix<TestType>(m, m);
        REQUIRE(tBLAS_test::Trivial<TestType>::transpose(A) == tBLAS::BLAS<TestType>::transpose(A));
    }
}

TEMPLATE_TEST_CASE("Transpose Rectangle Random", "[transpose]", int, float, double)
{
    using namespace std;
    for (int i = 0; i < 10; i++)
    {
        int m = GENERATE(take(1, random(50, 100)));
        int n = GENERATE(take(1, random(50, 100)));

        vector<vector<TestType>> A = genRandMatrix<TestType>(m, n);
        REQUIRE(tBLAS_test::Trivial<TestType>::transpose(A) == tBLAS::BLAS<TestType>::transpose(A));
    }
}