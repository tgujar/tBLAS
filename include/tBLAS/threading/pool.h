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
        /**
         * @brief A thread pool implementation.
         *
         */
        class ThreadPool
        {
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
            unsigned int num_threads;              /**< the number of threads in the pool */
            std::atomic<unsigned int> in_progress; /**< the number of jobs being executed or waiting to be executed */
            bool terminate_pool;                   /**< flag to indicate that the pool should be terminated */
            std::condition_variable jq_cv;         /**< condition variable to notify threads of new jobs / pool termination */
            std::mutex jq_mutex;                   /**< mutex to make job queue thread-safe */
            std::vector<std::thread> threads;      /**< vector of threads in the pool */
            std::queue<std::function<void()>> jq;  /**< job queue */

            void spin();
        };

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
        };

    }; // namespace threading
} // namespace tBLAS

#endif // POOL_HPP