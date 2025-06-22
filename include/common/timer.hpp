#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include "common/logger.hpp"
#include "cuda_runtime.h"

#include <chrono>
#include <ratio>
#include <string>

namespace NexLab {
    class Timer {
    public:
        using s = std::ratio<1, 1>;
        using ms = std::ratio<1, 1000>;
        using us = std::ratio<1, 1000000>;
        using ns = std::ratio<1, 1000000000>;
        /* data */
    public:
        Timer(/* args */) {
            time_elasped_ = 0;
            cpu_start_ = std::chrono::high_resolution_clock::now();
            cpu_stop_ = std::chrono::high_resolution_clock::now();

            cudaEventCreate(&gpu_start_);
            cudaEventCreate(&gpu_stop_);
        }

        ~Timer() {
            cudaFree(gpu_start_);
            cudaFree(gpu_stop_);
        }

        void start_cpu() {
            cpu_start_ = std::chrono::high_resolution_clock::now();
        }

        void start_gpu() {
            cudaEventRecord(gpu_start_, 0);
        }

        void stop_cpu() {
            cpu_stop_ = std::chrono::high_resolution_clock::now();
        }

        void stop_gpu() {
            cudaEventRecord(gpu_stop_, 0);
        }

        template <typename span>
        void duration_cpu(std::string msg) {
            std::string str;
            if (std::is_same<span, s>::value) {
                str = "s";
            } else if (std::is_same<span, s>::value) {
                str = "ms";
            } else if (std::is_same<span, s>::value) {
                str = "us";
            } else if (std::is_same<span, s>::value) {
                str = "ns";
            }

            std::chrono::duration<double, span> time = cpu_stop_ - cpu_start_;
            LOG_INFO(Logger::GetInstance(), "{} uses {} {}", msg.c_str(), time, str.c_str());
        }

        void duration_gpu(std::string msg) {
            CHECK_CUDA(cudaEventSynchronize(gpu_start_));
            CHECK_CUDA(cudaEventSynchronize(gpu_stop_));
            cudaEventElapsedTime(&time_elasped_, gpu_start_, gpu_stop_);

            LOG_INFO(Logger::GetInstance(), "{} uses {} {}", msg.c_str(), time_elasped_);
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start_;
        std::chrono::time_point<std::chrono::high_resolution_clock> cpu_stop_;
        cudaEvent_t gpu_start_;
        cudaEvent_t gpu_stop_;
        float time_elasped_;
    };
}; // namespace NexLab

#endif