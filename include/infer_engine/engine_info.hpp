#ifndef __ENGINE_INFO_HPP__
#define __ENGINE_INFO_HPP__
#include <stdint.h>

namespace NexLab {
    struct ioDims {
        /// @brief 能够处理的最大tensor维度
        static constexpr int32_t MAX_DIMS{8};
        /// @brief 实际tensor维度
        int32_t nbDims{-1};
        /// @brief 每个维度的实际tensor参数
        int64_t d[MAX_DIMS]{0};
        bool is_input = false;
    };
} // namespace NexLab

#endif