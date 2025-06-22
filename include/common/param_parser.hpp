#ifndef __PARAM_PARSER_HPP__
#define __PARAM_PARSER_HPP__
#include "Singleton.hpp"
#include "logger.hpp"
#include "to_json.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace NexLab {
    class ParamParser : public Singleton<ParamParser> {
    public:
        template <typename T>
        bool read_params(const std::string& file_name, T& params) {
            try {

                std::ifstream ifs(file_name);

                if (!ifs.is_open()) {
                    LOG_ERROR(Logger::GetInstance(), "Failed to open params file: {}", file_name);
                    return false;
                }

                nlohmann::json j;
                ifs >> j;
                params = j.get<T>();

                return true;
            } catch (const std::exception& e) {

                LOG_ERROR(Logger::GetInstance(), "Error reading JSON file: {}", e.what());

                return false;
            }
        }

        template <typename T>
        bool write_params(const std::string& file_name, const T& params) {
            try {
                std::ofstream ofs(file_name);

                if (!ofs.is_open()) {
                    LOG_ERROR(Logger::GetInstance(), "Failed to open params file: {}", file_name);
                    return false;
                }

                nlohmann::json j = params;
                ofs << j.dump(4); // 使用4个空格进行初始化
                return true;
            } catch (const std::exception& e) {

                LOG_ERROR(Logger::GetInstance(), "Error reading JSON file: {}", e.what());
            }
        }

    private:
        ParamParser() = default;

        ParamParser(const ParamParser&) = delete;

        ParamParser& operator=(const ParamParser&) = delete;

        // 声明singleton为友元类，允许访问私有构造函数
        friend class Singleton<ParamParser>;
    };

} // namespace NexLab

#endif