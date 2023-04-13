#ifndef POOL_HPP
#define POOL_HPP

#include <vector>
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <algorithm>
#include <atomic>

namespace tBLAS
{
    namespace threading
    {
        class ThreadPool
        {
        private:
            unsigned int m_num_threads;              /**< the number of threads in the pool */
            std::atomic<unsigned int> m_in_progress; /**< the number of jobs being executed or waiting to be executed */
            bool m_terminate_pool;                   /**< flag to indicate that the pool should be terminated */
            std::condition_variable m_jq_cv;         /**< condition variable to notify threads of new jobs / pool termination */
            std::mutex m_jq_mutex;                   /**< mutex to make job queue thread-safe */
            std::vector<std::thread> m_threads;      /**< vector of threads in the pool */
            std::queue<std::function<void()>> m_jq;  /**< job queue */

        public:
            /**
             * @brief Construct a new Thread Pool object with n threads.
             *
             * If n is larger than the number of aviailable threads,
             * the number of threads is set to the number of available threads.
             *
             * @param n The number of threads.
             */
            ThreadPool(unsigned int n);

            /**
             * @brief Destroy the Thread Pool object
             * Follows RAII, calls stop() on destruction.
             */
            virtual ~ThreadPool();

            /**
             * @brief Get the num threads in the thread pool.
             *
             * @return number of threads.
             */
            unsigned int get_num_threads() const;

            /**
             * @brief Enqueue a task to be executed by the thread pool.
             *
             * Adds a task to the queue and notifies a thread to execute it.
             * Task may not be executed immediately.
             *
             * @param task The task to be executed.
             */
            void enqueue(const std::function<void()> &task);

            /**
             * @brief Stop the thread pool.
             *
             * Calls join on all threads and waits for them to finish.
             *
             * @warning Does not execute tasks that are in the queue by not yet started.
             * @waning Thread pool is not reusable after calling this function.
             */
            void stop();

            /**
             * @brief Synchonize all threads in the pool.
             *
             * Waits for all threads to finish executing their tasks.
             *
             * @note This function does not call join() on the threads. Threads are reusable after calling this function.
             */
            void sync();

            /**
             * @brief Check if there are any jobs in the queue or any jobs being executed.
             */
            bool busy();

        private:
            /**
             * @brief Creates a thread and waits for jobs to be executed.
             *
             */
            void spin();

        }; // class ThreadPool

        /**
         * @brief A global thread pool.
         *
         * Provides access to a singleton ThreadPool object.
         * All tBLAS functions use this thread pool.
         *
         * @see ThreadPool
         *
         * @note In the current implementation, the global thread pool always has the same number
         * of threads as the number of available hardware threads.
         *
         */
        class GlobalThreadPool : public ThreadPool
        {
        public:
            GlobalThreadPool(const GlobalThreadPool &) = delete;
            GlobalThreadPool &operator=(const GlobalThreadPool &) = delete;

            /**
             * @brief Get the instance of the global thread pool.
             *
             * @return reference to thread pool.
             */
            static GlobalThreadPool &get_instance();

        private:
            /**
             * @brief Construct the Singleton Global Thread Pool object.
             *
             * The number of threads in the thread pool are equal to the number of hardware threads.
             */
            GlobalThreadPool();
        }; // class GlobalThreadPool

        /* ------------------------ ThreadPool Implementation ----------------------- */
        inline ThreadPool::ThreadPool(unsigned int n)
            : m_in_progress(0),
              m_terminate_pool(false),
              m_jq_cv(),
              m_jq_mutex(),
              m_threads(),
              m_jq()
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

        inline ThreadPool::~ThreadPool()
        {
            stop();
        }

        inline unsigned int ThreadPool::get_num_threads() const
        {
            return m_num_threads;
        }

        inline void ThreadPool::enqueue(const std::function<void()> &task)
        {
            {
                std::unique_lock<std::mutex> lk(m_jq_mutex);
                m_jq.push(task);
                m_in_progress++;
            }
            m_jq_cv.notify_one(); // notify one thread to execute the task
        }

        inline void ThreadPool::spin()
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

        inline void ThreadPool::sync()
        {
            std::unique_lock<std::mutex> lk(m_jq_mutex);
            m_jq_cv.wait(lk, [this]()
                         { return m_in_progress == 0; });
        }

        inline bool ThreadPool::busy()
        {
            bool poolbusy = true;
            {
                std::unique_lock<std::mutex> lk(m_jq_mutex);
                poolbusy = !(m_in_progress == 0);
            }
            return poolbusy;
        }

        inline void ThreadPool::stop()
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

        inline GlobalThreadPool::GlobalThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

        inline GlobalThreadPool &GlobalThreadPool::get_instance()
        {
            static GlobalThreadPool instance;
            return instance;
        }

    }; // namespace threading
};     // namespace tBLAS

#endif // POOL_HPP
