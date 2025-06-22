#ifndef __INFERX_CONFIG_HPP__
#define __INFERX_CONFIG_HPP__
#include "device_info.hpp"
#include "model_param.hpp"

#include <string>

namespace NexLab {
    enum class InferenceEngine {
        TensorRT, // NVIDIA TensorRT
        OpenVINO, // Intel OpenVINO
        ONNXRuntime, // Microsoft ONNX Runtime
        LibTorch, // PyTroch C++(LibTorch)
        OpenCV, // OpenCV DNN Module
        Unknown
    };

    struct InferXConfig {
        std::string model_path;
        std::string params_path;
        std::string dataset_path;
        InferenceEngine infer_engine;
        DeviceType dev_type;
        ModelType model_type;
        bool enable_visualization{true};
    };

}; // namespace NexLab

#endif