#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include "Singleton.hpp"

#include <chrono>
#include <cstdlib> // 用于获取进程 ID
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <NvInfer.h> // TensorRT 头文件
#include <cuda_runtime.h>
#define FMT_HEADER_ONLY
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h> // Windows 颜色常量
#endif
namespace NexLab {

    class Logger : public Singleton<Logger> {
    public:
        // 初始化日志系统
        void Init(const std::string& loggerName, const std::string& logFilePath = "") {
            try {
                // 初始化常规日志器
                if (!logFilePath.empty()) {
                    // 同时输出到控制台和文件
                    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, true);
                    logger =
                        std::make_shared<spdlog::logger>(loggerName, spdlog::sinks_init_list{console_sink, file_sink});
                } else {
                    // 仅输出到控制台
                    logger = spdlog::stdout_color_mt(loggerName);
                }

                // 设置常规日志格式（包含文件名和行号）
                logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [PID: %P] [TID: %t] [%s:%#] %v");

                // 初始化启动日志器（仅用于启动时）
                startup_logger = spdlog::stdout_color_mt("startup_logger");
                startup_logger->set_level(spdlog::level::info); // 设置为 info 级别，避免日志级别干扰
                startup_logger->set_pattern("%v"); // 简化输出格式，仅显示消息内容

                // 自定义颜色设置（兼容 Windows 和 Linux）
                auto color_sink = dynamic_cast<spdlog::sinks::stdout_color_sink_mt*>(logger->sinks()[0].get());
                if (color_sink) {
#ifdef _WIN32
                    // Windows 颜色设置
                    color_sink->set_color(spdlog::level::err, FOREGROUND_RED | FOREGROUND_INTENSITY); // 红色
                    color_sink->set_color(
                        spdlog::level::warn, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY); // 黄色
                    color_sink->set_color(spdlog::level::info, FOREGROUND_GREEN | FOREGROUND_INTENSITY); // 绿色
                    color_sink->set_color(
                        spdlog::level::critical, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY); // 紫色
#else
                    // Linux 颜色设置（ANSI 转义序列）
                    color_sink->set_color(spdlog::level::err, "\033[1;31m"); // 红色
                    color_sink->set_color(spdlog::level::warn, "\033[1;33m"); // 黄色
                    color_sink->set_color(spdlog::level::info, "\033[1;32m"); // 绿色
                    color_sink->set_color(spdlog::level::critical, "\033[1;35m"); // 紫色
#endif
                }

                // 注册日志器
                spdlog::register_logger(logger);
            } catch (const spdlog::spdlog_ex& ex) {
                std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
            }
        }

        // 动态加载效果函数
        void LoadingAnimation(const std::string& message, int duration_ms = 2000) {
            const char* spinner[] = {"|", "/", "-", "\\"};
            auto end_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
            while (std::chrono::steady_clock::now() < end_time) {
                for (const auto& symbol : spinner) {
#ifdef _WIN32
                    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_INTENSITY);
                    startup_logger->info("{} {}", message, symbol);
                    SetConsoleTextAttribute(
                        GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
#else
                    startup_logger->info("\033[1;36m{} {}\033[0m", message, symbol); // 青色加载动画
#endif
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            }
            startup_logger->info("{}{}", std::string(message.size() + 2, ' '), "\r");
        }

        // 显示启动日志
        void ShowStartupScreen() {
#ifdef _WIN32
            // Windows 颜色常量
            const int title_color = FOREGROUND_BLUE | FOREGROUND_INTENSITY; // 克莱因蓝（深蓝色 + 高亮）
            const int text_color = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY; // 白色
            const int reset_color = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; // 重置颜色
#else
            // Linux ANSI 颜色
            const std::string title_color = "\033[1;34m"; // 克莱因蓝（使用 ANSI 蓝色 + 高亮）
            const std::string text_color = "\033[1;37m"; // 白色
            const std::string reset_color = "\033[0m"; // 重置颜色
#endif

            // 使用启动日志器输出启动界面
#ifdef _WIN32
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), title_color);
            // 使用启动日志器输出启动界面
            startup_logger->info(R"(
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   ██╗███╗   ██╗███████╗██████╗ ██████╗    ██╗  ██╗                 │
    │   ██║████╗  ██║██╔════╝██╔═══╝ ██╔══██╗   ╚██╗██╔╝                 │
    │   ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝    ╚███╔╝                  │
    │   ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗    ██╔██╗                  │
    │   ██║██║ ╚████║██║     ███████╗██║  ██║   ██╔╝ ██╗                 │
    │   ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝  ╚═╝                 │
    │                                                                    │
    │   Infer-X - 高性能推理引擎，致力于为 AI 应用提供极致性能。         │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    )");
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), reset_color);
#else
            startup_logger->info(R"(
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   ██╗███╗   ██╗███████╗██████╗ ██████╗    ██╗  ██╗                 │
    │   ██║████╗  ██║██╔════╝██╔═══╝ ██╔══██╗   ╚██╗██╔╝                 │
    │   ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝    ╚███╔╝                  │
    │   ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗    ██╔██╗                  │
    │   ██║██║ ╚████║██║     ███████╗██║  ██║   ██╔╝ ██╗                 │
    │   ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝  ╚═╝                 │
    │                                                                    │
    │   Infer-X - 高性能推理引擎，致力于为 AI 应用提供极致性能。         │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    )",
                title_color, reset_color, title_color, reset_color, title_color, reset_color, title_color, reset_color,
                title_color, reset_color, title_color, reset_color, text_color, reset_color);
#endif
        }

        // 记录普通日志（带文件名和行号）
        template <typename... Args>
        void Log(
            spdlog::level::level_enum level, const char* file, int line, const std::string& message, Args&&... args) {
            if (logger) {
                logger->log(
                    spdlog::source_loc{file, line, SPDLOG_FUNCTION}, level, message, std::forward<Args>(args)...);
            } else {
                std::cerr << "Logger not initialized!" << std::endl;
            }
        }

        // TensorRT 日志接口（不打印文件名和行号）
        void LogTRT(nvinfer1::ILogger::Severity severity, const char* msg) {
            if (!logger) {
                std::cerr << "Logger not initialized!" << std::endl;
                return;
            }

            // logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [PID: %P] [TID: %t] %v");

            switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                logger->log(spdlog::level::critical, "[TensorRT] INTERNAL_ERROR: {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                logger->log(spdlog::level::err, "[TensorRT] ERROR: {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                logger->log(spdlog::level::warn, "[TensorRT] WARNING: {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                logger->log(spdlog::level::info, "[TensorRT] INFO: {}", msg);
                break;
            default:
                logger->log(spdlog::level::debug, "[TensorRT] VERBOSE: {}", msg);
                break;
            }
        }

    private:
        std::shared_ptr<spdlog::logger> logger; // 常规日志器
        std::shared_ptr<spdlog::logger> startup_logger; // 启动日志器
    };

    class TRTLogger : public nvinfer1::ILogger {
    public:
        TRTLogger() = default;
        virtual ~TRTLogger() = default;
        void log(Severity severity, const char* msg) noexcept override {
            Logger::GetInstance().LogTRT(severity, msg);
        }
    };


    inline void init_logger() {
        static bool initialized = false;

        if (!initialized) {
            try {
#ifdef _WIN32
                SetConsoleOutputCP(CP_UTF8); // 设置控制台输出为 UTF-8 编码
#endif
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                std::tm tm = *std::localtime(&time);
                std::ostringstream oss;
                oss << std::put_time(&tm, "%Y%m%d");

                std::string project_root = std::filesystem::current_path().string();
                std::string log_dir = project_root + "./logs";

                if (!std::filesystem::exists(log_dir)) {
                    std::filesystem::create_directory(log_dir);
                }
                std::string log_path = log_dir + "/inferx-log-" + oss.str() + ".log";

                Logger::GetInstance().Init("infer-X", log_path);
                Logger::GetInstance().ShowStartupScreen();

                initialized = true;
            } catch (const std::exception& e) {
                std::cerr << "CRITICAL: " << e.what() << std::endl;
                std::terminate(); // 或其它错误处理
            }
        }
    }

} // namespace NexLab

// 定义宏简化日志调用
#define LOG_INFO(logger, ...)     logger.Log(spdlog::level::info, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(logger, ...)  logger.Log(spdlog::level::warn, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(logger, ...)    logger.Log(spdlog::level::err, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_CRITICAL(logger, ...) logger.Log(spdlog::level::critical, __FILE__, __LINE__, __VAArgs__)

#define CHECK_CUDA(call)                                                                         \
    do {                                                                                         \
        cudaError_t err = call;                                                                  \
        if (err != cudaSuccess) {                                                                \
            LOG_ERROR(NexLab::Logger::GetInstance(), "CUDA error: {}", cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error");                                              \
        }                                                                                        \
    } while (0)


#endif // __LOGGER_HPP__