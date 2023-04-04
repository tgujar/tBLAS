#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include "trivial.hpp"

using namespace std;

TEST_CASE("Transpose trivial", "[transpose]")
{
    vector<vector<int>> C = {{1, 2, 3}, {4, 5, 6}};
    REQUIRE(tBLAS_test::Trivial<int>().transpose(C) == vector<vector<int>>{{1, 4}, {2, 5}, {3, 6}});
}