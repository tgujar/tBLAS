#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include "trivial.hpp"

using namespace std;

TEST_CASE("Matmul trivial", "[matmul]")
{
    vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
    vector<vector<int>> B = {{1, 2}, {3, 4}, {5, 6}};
    REQUIRE(tBLAS_test::Trivial<int>().matmul(A, B) == vector<vector<int>>{{22, 28}, {49, 64}});
}