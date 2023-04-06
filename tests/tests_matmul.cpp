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

TEST_CASE("Matmul trivial", "[matmul]")
{
    SECTION("Square matrices")
    {
        using namespace std;
        array<array<int, 2>, 2> A = {{{1, 2}, {3, 4}}};
        array<array<int, 2>, 2> B = {{{1, 2}, {3, 4}}};
        REQUIRE(tBLAS_test::Trivial::matmul(A, B) == array<array<int, 2>, 2>{{{7, 10}, {15, 22}}});
    }
    SECTION("Rectangular matrices")
    {
        using namespace std;
        array<array<int, 3>, 2> A = {{{1, 2, 3}, {4, 5, 6}}};
        array<array<int, 2>, 3> B = {{{1, 2}, {3, 4}, {5, 6}}};
        REQUIRE(tBLAS_test::Trivial::matmul(A, B) == array<array<int, 2>, 2>{{{22, 28}, {49, 64}}});
    }
}

// TEMPLATE_TEST_CASE("Matmul Square Random Int", "[matmul]", int)
// {
//     using namespace std;
//     for (int i = 0; i < 10; i++)
//     {
//         size_t m = GENERATE(take(1, random(10, 50)));

//         array<array<TestType, 2>, 2> A = gen_rand_matrix<TestType, m, m>();
//         array<array<TestType, 2>, 2> B = gen_rand_matrix<TestType, m, m>();

//         REQUIRE(tBLAS_test::Trivial::matmul(A, B) == tBLAS::BLAS::matmul(A, B));
//     }
// }

// TEST_CASE("Matmul Rectangle Random Int", "[matmul]")
// {
//     using namespace std;

//     for (int i = 0; i < 10; i++)
//     {
//         size_t m = GENERATE(take(1, random(10, 50)));
//         size_t n = GENERATE(take(1, random(10, 50)));
//         size_t p = GENERATE(take(1, random(10, 50)));

//         vector<vector<int>> A = gen_rand_matrix<int>(m, n);
//         vector<vector<int>> B = gen_rand_matrix<int>(n, p);

//         tBLAS::Matrix<int, m, n> tA(A);

//         compare_INT_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::BLAS::matmul(A, B).to_vector());
//     }
// }

// // TEMPLATE_TEST_CASE("Matmul Rectangle Random FP", "[matmul]", float, double)
// // {
// //     using namespace std;

// //     for (int i = 0; i < 10; i++)
// //     {
// //         int m = GENERATE(take(1, random(10, 50)));
// //         int n = GENERATE(take(1, random(10, 50)));
// //         int p = GENERATE(take(1, random(10, 50)));

// //         vector<vector<TestType>> A = gen_rand_matrix<TestType>(m, n);
// //         vector<vector<TestType>> B = gen_rand_matrix<TestType>(n, p);

// //         compare_FP_2D(tBLAS_test::Trivial::matmul(A, B), tBLAS::BLAS::matmul(A, B));
// //     }
// // }

TEST_CASE("Matmul INT Square Identity", "[matmul]")
{
    using namespace std;
    tBLAS::Matrix<int, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    tBLAS::Matrix<int, 3, 3> B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    std::vector<std::vector<int>> v_A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<int>> v_B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    compare_INT_2D(tBLAS_test::Trivial::matmul(v_A, v_B), tBLAS::BLAS::matmul(A, B).to_vector());
}

TEST_CASE("Matmul INT Rectangle Identity", "[matmul]")
{
    using namespace std;
    tBLAS::Matrix<int, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
    tBLAS::Matrix<int, 3, 2> B{{1, 2}, {3, 4}, {5, 6}};

    std::vector<std::vector<int>> v_A{{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> v_B{{1, 2}, {3, 4}, {5, 6}};
    compare_INT_2D(tBLAS_test::Trivial::matmul(v_A, v_B), tBLAS::BLAS::matmul(A, B).to_vector());
}
