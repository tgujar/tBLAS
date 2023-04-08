#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include <vector>
#include <thread>

#include <tBLAS/matrix.hpp>
#include <tBLAS/threading/pool.h>
#include "trivial.hpp"
#include "helpers.hpp"

TEST_CASE("Thread pool", "[matrix]")
{
    using namespace std;
    using namespace tBLAS::threading;
    SECTION("Initialization")
    {
        ThreadPool pool(std::thread::hardware_concurrency());
        INFO("HW threads: " + to_string(std::thread::hardware_concurrency()));
        REQUIRE(pool.get_num_threads() == std::thread::hardware_concurrency());
    }

    SECTION("Counter")
    {
        ThreadPool pool(std::thread::hardware_concurrency());
        for (int i = 0; i < 10; i++)
        {

            for (int i = 0; i < 10; i++)
            {
                pool.enqueue([i]()
                             { "Thread" + to_string(i); });
            }
            pool.sync();
        }
        pool.stop();
        WARN("okay");
    }
}