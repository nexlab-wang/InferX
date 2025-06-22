#ifndef __TENSORRT_ENGINE_HPP__
#define __TENSORRT_ENGINE_HPP__
#include "NvInfer.h"
#include "base_engine.hpp"

#include <iostream>
#include <memory>

namespace NexLab {
    class MODEL_INFER_API TensorRTEngine : public BaseEngine {
    public:
        TensorRTEngine();
        
        ~TensorRTEngine() override;

        bool load_dev();

        bool load_model(const std::string& model) override;

        bool infer_model() override;

        void checking() override;

        bool set_dynamic_shapes();

        const std::unordered_map<std::string, ioDims>& get_io_dims() const override;

    private:
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        cudaStream_t stream_;

        std::unordered_map<std::string, ioDims> io_dims_map_;

        int32_t tensor_io_nums_ = -1;
        std::vector<const char*> tensor_io_names_;
        std::vector<nvinfer1::Dims> tensor_io_dims_;
    };

} // namespace NexLab

#endif