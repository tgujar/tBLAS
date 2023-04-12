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
        ThreadPool::ThreadPool(unsigned int n) : m_in_progress(0), m_terminate_pool(false), m_jq_cv(), m_jq_mutex(), m_threads(), m_jq()
        {
            unsigned int hw_threads = std::thread::hardware_concurrency();

            // use atleast 4 threads if hardware_concurrency() not available
            m_num_threads = std::min(hw_threads == 0 ? 4 : hw_threads, n);
            m_threads.reserve(m_num_threads);

            for (unsigned int i = 0; i < m_num_threads; i++)
            {
                m_threads.push_back(std::thread([this]
                                                { spin(); }));
            }
        }

        ThreadPool::~ThreadPool()
        {
            stop();
        }

        unsigned int ThreadPool::get_num_threads() const
        {
            return m_num_threads;
        }

        void ThreadPool::enqueue(const std::function<void()> &task)
        {
            {
                std::unique_lock<std::mutex> lk(m_jq_mutex);
                m_jq.push(task);
                m_in_progress++;
            }
            m_jq_cv.notify_one(); // notify one thread to execute the task
        }

        void ThreadPool::spin()
        {
            while (true)
            {
                std::function<void()> curr_job;
                {
                    std::unique_lock<std::mutex> lk(m_jq_mutex);

                    // nothing to do if task queue is empty and pool is not being terminated
                    m_jq_cv.wait(lk, [this]()
                                 { return !m_jq.empty() || m_terminate_pool; });

                    if (m_terminate_pool)
                    {
                        return;
                    }
                    curr_job = m_jq.front();
                    m_jq.pop();
                }

                curr_job();

                if (--m_in_progress == 0) // task completed
                {
                    m_jq_cv.notify_all(); // wake up sync() if its waiting
                }
            }
        }

        void ThreadPool::sync()
        {
            std::unique_lock<std::mutex> lk(m_jq_mutex);
            m_jq_cv.wait(lk, [this]()
                         { return m_in_progress == 0; });
        }

        bool ThreadPool::busy()
        {
            bool poolbusy = true;
            {
                std::unique_lock<std::mutex> lk(m_jq_mutex);
                poolbusy = !(m_in_progress == 0);
            }
            return poolbusy;
        }

        void ThreadPool::stop()
        {
            {
                std::unique_lock<std::mutex> lk(m_jq_mutex);
                m_terminate_pool = true;
            }
            m_jq_cv.notify_all();
            for (auto &t : m_threads)
            {
                t.join();
            }
            m_threads.clear();
        }

        /* ------------------------ GlobalThreadPool Implementation ----------------------- */

        GlobalThreadPool::GlobalThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

        GlobalThreadPool &GlobalThreadPool::get_instance()
        {
            static GlobalThreadPool instance;
            return instance;
        }

    }; // namespace threading
};     // namespace tBLAS