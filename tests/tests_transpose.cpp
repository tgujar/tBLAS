#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/blas.hpp>
#include "trivial.hpp"
#include "helpers.hpp"

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
