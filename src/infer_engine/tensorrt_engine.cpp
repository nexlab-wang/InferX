#include "infer_engine/tensorrt_engine.hpp"

#include "common/utils.hpp"

#include <sstream>
namespace NexLab {
    TRTLogger trt_logger;

    TensorRTEngine::TensorRTEngine() : runtime_(nullptr), engine_(nullptr), context_(nullptr), stream_(nullptr) {
        LOG_INFO(Logger::GetInstance(), "Starting Inference with TensorRT Engine...");
    }

    TensorRTEngine::~TensorRTEngine() {
        LOG_INFO(Logger::GetInstance(), "TensorRT Engine is being deleted.");
    }

    bool TensorRTEngine::load_dev() {
        // 只支持GPU设备(这里可能需要读取参数配置)
        auto& dev_info = get_device_info();
        cudaSetDevice(dev_info.get_dev_id());

        // 检查流是否为 CUDA_INVALID_STREAM
        if (stream_ == cudaStreamLegacy || stream_ == cudaStreamPerThread) {
            return true;
        }

        // 检查流是否在地址空间内
        if (stream_ == nullptr) {
            cudaStreamCreate(&stream_);

            // 尝试使用流，如果失败则流无效
            cudaError_t error = cudaStreamQuery(stream_);
            if (error != cudaSuccess && error != cudaErrorNotReady) {
                return false;
            }
        }

        return true;
    }

    bool TensorRTEngine::load_model(const std::string& model) {

        if (model.empty()) {
            LOG_WARNING(Logger::GetInstance(), "No model file path has been specified!");
            return false;
        }

        std::vector<unsigned char> trt_file = utils::loadModel(model);
        if (trt_file.empty()) {
            LOG_WARNING(Logger::GetInstance(), "Model file: {} is empty!", model);
            return false;
        }

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger));
        if (!runtime_) {
            // todo add log
            return false;
        }

        engine_ =
            std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(trt_file.data(), trt_file.size()));
        if (!engine_) {
            // todo add log
            return false;
        }

        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            // todo add log
            return false;
        }

        checking();
        return true;
    }

    bool TensorRTEngine::infer_model() {

        auto input_datas = get_input_datas();
        auto output_datas = get_output_datas();

        std::vector<float*> bindings;
        bindings.insert(bindings.end(), input_datas.begin(), input_datas.end());
        bindings.insert(bindings.end(), output_datas.begin(), output_datas.end());

        bool context = context_->executeV2((void**) bindings.data());
        cudaDeviceSynchronize();
        return context;
    }

    void TensorRTEngine::checking() {
        tensor_io_nums_ = engine_->getNbIOTensors();
        io_dims_map_.clear();

        for (int i = 0; i < tensor_io_nums_; ++i) {

            auto tensor_io_name = engine_->getIOTensorName(i);
            // nvinfer1::DataType type = engine_->getTensorDataType(tensor_io_name);
            // size_t element_size = utils::GetDataTypeSize(type);
            // size_t buffer_size; // 计算buffer
            auto dims = engine_->getTensorShape(tensor_io_name);

            ioDims io_dims;
            std::string tensor_io_name_str = tensor_io_name;
            io_dims.nbDims = dims.nbDims;
            io_dims.is_input = engine_->getTensorIOMode(tensor_io_name) == nvinfer1::TensorIOMode::kINPUT ? 1 : 0;

            for (int i = 0; i < dims.nbDims; ++i) {
                io_dims.d[i] = dims.d[i] == -1 ? b_image_params_->batch_size : dims.d[i];
            }

            io_dims_map_.insert(std::make_pair(tensor_io_name_str, io_dims));

            std::ostringstream engine_info;
            engine_info << "The engine's info: idx = " << i << ", " << tensor_io_name << ": ";

            for (int j = 0; j < dims.nbDims; j++) {
                engine_info << ", ";
                engine_info << io_dims.d[j];
            }

            LOG_INFO(Logger::GetInstance(), "{}", engine_info.str());
        }

        if (b_image_params_->dynamic_input == true) {
            if (!set_dynamic_shapes())
                LOG_ERROR(Logger::GetInstance(), "Set dynamic batch is Failed");
        }
    }

    bool TensorRTEngine::set_dynamic_shapes() {
        if (!context_)
            return false;

        for (const auto& [name, dims] : io_dims_map_) {
            if (!dims.is_input)
                continue;

            nvinfer1::Dims shape;
            shape.nbDims = dims.nbDims;
            for (int i = 0; i < dims.nbDims; ++i) {
                shape.d[i] = dims.d[i];
            }


            if (!context_->setInputShape(name.c_str(), shape)) {
                LOG_ERROR(Logger::GetInstance(), "Failed to set shape for input: {}", name);
                return false;
            }
        }

        return context_->allInputDimensionsSpecified();
    }

    const std::unordered_map<std::string, ioDims>& TensorRTEngine::get_io_dims() const {
        return io_dims_map_;
    }
} // namespace NexLab
