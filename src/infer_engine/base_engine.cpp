#include "infer_engine/base_engine.hpp"

namespace NexLab {
    void BaseEngine::set_device_info(const DeviceInfo& dev_info) {
        dev_info_ = dev_info;
    }

    const DeviceInfo& BaseEngine::get_device_info() const {
        return dev_info_;
    }

    void BaseEngine::set_input_datas(std::vector<float*> inputs) {
        input_src_datas_ = inputs;
    }

    const std::vector<float*> BaseEngine::get_input_datas() const {
        return input_src_datas_;
    }

    void BaseEngine::set_output_datas(std::vector<float*> outputs) {
        output_src_datas_ = outputs;
    }

    const std::vector<float*> BaseEngine::get_output_datas() const {
        return output_src_datas_;
    }

    void BaseEngine::initialize(const ModelParamsBase& params) {
        params.validate();
        this->b_params_ = std::make_shared<const ModelParamsBase>(params);

        switch (params.model_type) {
        case ModelType::YOLO_DETECTION:
        case ModelType::YOLO_POSE:
        case ModelType::YOLO_SEGMENT:
        case ModelType::YOLO_OBB:
        case ModelType::RT_DETR:
            {
                auto image_params = dynamic_cast<const ImageModelParams*>(&params);
                if (!image_params) {
                    throw std::runtime_error("Invalid model parameters: expected ImageModelParams");
                }

                this->b_image_params_ = std::make_shared<const ImageModelParams>(*image_params);

                if (b_image_params_) {
                    const char* model_name = (params.model_type == ModelType::RT_DETR) ? "RT_DETR" : "YOLO";

                    LOG_INFO(Logger::GetInstance(), "Initializing {} with input dimensions: {} x {}", model_name,
                        b_image_params_->dst_h, b_image_params_->dst_w);
                }
                break;
            }
        case ModelType::POINTNET:
            {
                LOG_INFO(Logger::GetInstance(), "Codeing...");
                break;
            }
        default:
            throw std::runtime_error("Unsupported model type: " + std::to_string(static_cast<int>(params.model_type)));
        }
    }
} // namespace NexLab
