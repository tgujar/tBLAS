#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/matrix.hpp>
#include "trivial.hpp"
#include "helpers.hpp"

TEST_CASE("Matrix Basic", "[matrix]")
{
    SECTION("Initialization")
    {
        std::vector<std::vector<int>> vals{{1, 2, 3, 4}, {5, 6, 7, 8}};
        tBLAS::MatrixXd<int, 2, 4> A(vals);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                INFO(A(i, j));
                REQUIRE(A(i, j) == (i * 4 + j + 1));
            }
        }
    }
}