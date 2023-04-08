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
        public:
            ThreadPool(unsigned int n);
            virtual ~ThreadPool();

            unsigned int get_num_threads() const;
            void enqueue(const std::function<void()> &task);
            void stop();
            void sync();
            bool busy();

        private:
            unsigned int num_threads;
            std::atomic<unsigned int> in_progress;
            void spin();
            bool terminate_pool;
            std::condition_variable jq_cv;
            std::mutex jq_mutex;
            std::vector<std::thread> threads;
            std::queue<std::function<void()>> jq;
        };

        class GlobalThreadPool : public ThreadPool
        {
        public:
            GlobalThreadPool(const GlobalThreadPool &) = delete;
            GlobalThreadPool &operator=(const GlobalThreadPool &) = delete;
            static GlobalThreadPool &get_instance();

        private:
            GlobalThreadPool();
        };

    }; // namespace threading
} // namespace tBLAS

#endif // POOL_HPP