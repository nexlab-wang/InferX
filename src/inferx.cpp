#include "inferx.hpp"

namespace NexLab {

    InferX::InferX() : engine_(nullptr), model_(nullptr), model_params_(nullptr) {
        init_logger();
    }

    bool InferX::load_config(std::string config_file_path) {
        ParamParser& parser = ParamParser::GetInstance();

        if (!parser.read_params(config_file_path, infer_config_)) {
            LOG_ERROR(Logger::GetInstance(), "Failed to read InferX Config from file.");
            return false;
        }
        LOG_INFO(Logger::GetInstance(), "InferX Config file read successfully!");

        auto params = std::make_shared<NexLab::ImageModelParams>();
        if (!parser.read_params(infer_config_.params_path, *params)) {
            LOG_ERROR(Logger::GetInstance(), "Failed to read model parameters.");
            return false;
        }
        LOG_INFO(Logger::GetInstance(), "Model Parameters read successfully!");
        model_params_ = params;

        try {
            switch (infer_config_.infer_engine) {
            case InferenceEngine::TensorRT:
                engine_ = std::make_shared<TensorRTEngine>();
                break;
            case InferenceEngine::OpenVINO:
                engine_ = std::make_shared<OpenVINOEngine>();
                break;
            default:
                throw std::runtime_error("Unsupported inference engine");
            }

            DeviceInfo device(infer_config_.dev_type, 0);
            engine_->set_device_info(device);
        } catch (const std::exception& e) {
            LOG_ERROR(Logger::GetInstance(), "Engine Initalization failed: " + std::string(e.what()));
            return false;
        }

        // 这里参数应该传入的是ModelParamsBase基类，测试通过之后进行修改
        try {
            switch (infer_config_.model_type) {
            case ModelType::YOLO_DETECTION:
            case ModelType::YOLO_POSE:
            case ModelType::YOLO_SEGMENT:
            case ModelType::YOLO_OBB:
                model_ = std::make_shared<YOLO>(engine_, params);
                break;
            case ModelType::RT_DETR:
                model_ = std::make_shared<RTDETR>(engine_, params);
                break;
            default:
                throw std::runtime_error("Unsupported model type");
            }
        } catch (const std::exception& e) {
            LOG_ERROR(Logger::GetInstance(), "Model Initialization failed: " + std::string(e.what()));
            return false;
        }

        if (!model_->load_model(infer_config_.model_path)) {
            LOG_ERROR(Logger::GetInstance(), "Failed to load model: " + infer_config_.model_path);
            return false;
        }
        LOG_INFO(Logger::GetInstance(), "Model loaded successfully: " + infer_config_.model_path);

        return true;
    }

    bool InferX::init_dataset_stream() {
        close_stream();

        if (infer_config_.dataset_path.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Dataset file path is empty!");
            return false;
        }
        dataset_files_ = utils::load_dataset_list(infer_config_.dataset_path);
        current_dataset_index_ = 0;
        is_camera_stream_ = false;
        is_video_stream_ = false;
        return true;
    }

    bool InferX::get_next_frame(cv::Mat& frame) {
        if (!dataset_files_.empty()) {
            if (current_dataset_index_ >= dataset_files_.size())
                return false;
        }

        frame = cv::imread(dataset_files_[current_dataset_index_]);
        current_dataset_index_++;
        return !frame.empty();
    }

    bool InferX::model_infer(std::vector<cv::Mat>& inputs, std::vector<std::vector<InferRes>>& outputs) {
        if (!engine_ && !model_)
            return false;
        outputs = model_->detection(inputs);
    }

    bool InferX::is_stream_open() const {

        if (!dataset_files_.empty()) {
            return current_dataset_index_ < dataset_files_.size();
        }
        return false;
    }

    void InferX::close_stream() {
        is_camera_stream_ = false;
        is_video_stream_ = false;
        dataset_files_.clear();
        current_dataset_index_ = 0;
    }
} // namespace NexLab