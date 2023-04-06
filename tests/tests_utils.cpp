#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <tBLAS/utils.hpp>
#include <tBLAS/constants.hpp>
#include "trivial.hpp"
#include "helpers.hpp"

// TEST_CASE("Pack horizontal", "[pack]")
// {

//     SECTION("Square matrices")
//     {
//         using namespace std;
//         tBLAS::MatrixXd<int, tBLAS::KERNEL_KC, tBLAS::KERNEL_NR> M(gen_rand_matrix<int>(tBLAS::KERNEL_KC, tBLAS::KERNEL_NR));
//         auto packed = tBLAS::pack_horizontal(M, {tBLAS::KERNEL_KC, tBLAS::KERNEL_NR}, {0, 0});
//         for (int i = 0; i < 3; i++)
//         {
//             for (int j = 0; j < 3; j++)
//             {
//                 REQUIRE(packed(i, j) == M(i, j));
//             }
//         }
//     }
// }

// TEST_CASE("Pack vertical", "[pack]")
// {

//     SECTION("Square matrices")
//     {
//         using namespace std;
//         tBLAS::MatrixXd<int, tBLAS::KERNEL_KC, tBLAS::KERNEL_MR> M(gen_rand_matrix<int>(tBLAS::KERNEL_KC, tBLAS::KERNEL_MR));
//         auto packed = tBLAS::pack_vertical(M, {tBLAS::KERNEL_KC, tBLAS::KERNEL_MR}, {0, 0});
//         for (int i = 0; i < 3; i++)
//         {
//             for (int j = 0; j < 3; j++)
//             {
//                 REQUIRE(packed(i, j) == M(j, i));
//             }
//         }
//     }
// }

// TEMPLATE_TEST_CASE("Transpose Square Random", "[transpose]", int)
// {

//     using namespace std;
//     for (int i = 0; i < 10; i++)
//     {
//         int m = GENERATE(take(1, random(50, 100)));

//         vector<vector<TestType>> A = gen_rand_matrix<TestType>(m, m);
//         REQUIRE(tBLAS_test::Trivial::transpose(A) == tBLAS::BLAS::transpose(A));
//     }
// }
