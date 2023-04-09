#include <vector>
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <algorithm>

#include "./pool.h"

namespace tBLAS
{
    namespace threading
    {
        /* ------------------------ ThreadPool Implementation ----------------------- */
        ThreadPool::ThreadPool(unsigned int n) : in_progress(0), terminate_pool(false), jq_cv(), jq_mutex(), threads(), jq()
        {
            unsigned int hw_threads = std::thread::hardware_concurrency();

            // use atleast 4 threads if hardware_concurrency() not available
            num_threads = std::min(hw_threads == 0 ? 4 : hw_threads, n);
            threads.reserve(num_threads);

            for (unsigned int i = 0; i < num_threads; i++)
            {
                threads.push_back(std::thread([this]
                                              { spin(); }));
            }
        }

        ThreadPool::~ThreadPool()
        {
            stop();
        }

        unsigned int ThreadPool::get_num_threads() const
        {
            return num_threads;
        }

        void ThreadPool::enqueue(const std::function<void()> &task)
        {
            {
                std::unique_lock<std::mutex> lk(jq_mutex);
                jq.push(task);
                in_progress++;
            }
            jq_cv.notify_one(); // notify one thread to execute the task
        }

        void ThreadPool::spin()
        {
            while (true)
            {
                std::function<void()> curr_job;
                {
                    std::unique_lock<std::mutex> lk(jq_mutex);

                    // nothing to do if task queue is empty and pool is not being terminated
                    jq_cv.wait(lk, [this]()
                               { return !jq.empty() || terminate_pool; });

                    if (terminate_pool)
                    {
                        return;
                    }
                    curr_job = jq.front();
                    jq.pop();
                }

                curr_job();

                if (--in_progress == 0) // task completed
                {
                    jq_cv.notify_all(); // wake up sync() if its waiting
                }
            }
        }

        void ThreadPool::sync()
        {
            std::unique_lock<std::mutex> lk(jq_mutex);
            jq_cv.wait(lk, [this]()
                       { return in_progress == 0; });
        }

        bool ThreadPool::busy()
        {
            bool poolbusy = true;
            {
                std::unique_lock<std::mutex> lk(jq_mutex);
                poolbusy = !(in_progress == 0);
            }
            return poolbusy;
        }

        void ThreadPool::stop()
        {
            {
                std::unique_lock<std::mutex> lk(jq_mutex);
                terminate_pool = true;
            }
            jq_cv.notify_all();
            for (auto &t : threads)
            {
                t.join();
            }
            threads.clear();
        }

        /* ------------------------ GlobalThreadPool Implementation ----------------------- */
        GlobalThreadPool &GlobalThreadPool::get_instance()
        {
            static GlobalThreadPool instance;
            return instance;
        }

        GlobalThreadPool::GlobalThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

    }; // namespace threading
};     // namespace tBLAS