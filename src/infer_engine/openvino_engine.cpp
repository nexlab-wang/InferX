#include "infer_engine/openvino_engine.hpp"

namespace NexLab {
    OpenVINOEngine::OpenVINOEngine() : model_(nullptr), compiled_model_(nullptr), infer_request_(nullptr) {
        core_ = std::make_shared<ov::Core>();
        LOG_INFO(Logger::GetInstance(), "Starting Inference with OpenVINO Engine...");
    }

    OpenVINOEngine::~OpenVINOEngine() {
        LOG_INFO(Logger::GetInstance(), "OpenVINO Engine is being deleted.");
    }

    bool OpenVINOEngine::load_dev() {
        auto& dev_info = get_device_info();
        // 进行设备检查
        return true;
    }

    bool OpenVINOEngine::load_model(const std::string& model) {
        model_ = core_->read_model(model);

        // if (b_image_params_->dynamic_input) {
        //     if (!set_dynamic_shapes()) {
        //         LOG_ERROR(Logger::GetInstance(), "Set dynamic shape is Failed.");
        //     }
        // }

        compiled_model_ = std::make_shared<ov::CompiledModel>(core_->compile_model(model_, "CPU"));
        if (!compiled_model_) {
            return false;
        }

        infer_request_ = std::make_shared<ov::InferRequest>(compiled_model_->create_infer_request());
        if (!infer_request_) {
            return false;
        }

        checking();
        return true;
    }

    bool OpenVINOEngine::infer_model() {

        try {
            auto input_datas = get_input_datas();
            auto input_ports = compiled_model_->inputs();

            for (size_t i = 0; i < input_ports.size(); ++i) {

                std::string input_name = input_ports[i].get_any_name();
                auto input_dims = io_dims_map_.find(input_name);
                if (input_dims != io_dims_map_.end()) {
                    ov::Shape input_shape(input_dims->second.nbDims);

                    for (size_t i = 0; i < input_dims->second.nbDims; ++i) {
                        input_shape[i] = input_dims->second.d[i];
                    }

                    ov::Tensor input_tensor(input_ports[i].get_element_type(), input_shape, input_datas[i]);
                    infer_request_->set_input_tensor(i, input_tensor);
                } else {
                    LOG_ERROR(Logger::GetInstance(), "Model infer error");
                    return false;
                }
            }

            infer_request_->infer();

            auto output_datas = get_output_datas();
            auto output_ports = compiled_model_->outputs();

            for (size_t i = 0; i < output_ports.size(); ++i) {
                ov::Tensor output_tensor = infer_request_->get_output_tensor(i);
                float* output_data = output_tensor.data<float>();
                size_t output_size = output_tensor.get_size();

                std::memcpy(output_datas[i], output_data, output_size * sizeof(float));
            }
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR(Logger::GetInstance(), "Inference failed: {}", e.what());
            return false;
        }
    }

    void OpenVINOEngine::checking() {

        io_dims_map_.clear();

        for (const auto& input : model_->inputs()) {
            ioDims dims;
            auto pshape = input.get_partial_shape();
            dims.nbDims = pshape.rank().get_length();
            dims.is_input = true;

            for (int i = 0; i < dims.nbDims; ++i) {
                dims.d[i] = pshape[i].is_dynamic() ? b_image_params_->batch_size : pshape[i].get_length();
            }

            io_dims_map_[input.get_any_name()] = dims;

            std::ostringstream engine_info;

            for (int j = 0; j < dims.nbDims; j++) {
                engine_info << ", ";
                engine_info << dims.d[j];
            }

            LOG_INFO(Logger::GetInstance(), "Input: {}, shape: {}", input.get_any_name(), engine_info.str());
        }

        for (const auto& output : model_->outputs()) {
            ioDims dims;
            auto pshape = output.get_partial_shape();
            dims.nbDims = pshape.rank().get_length();
            dims.is_input = false;

            for (int i = 0; i < dims.nbDims; ++i) {
                dims.d[i] = pshape[i].is_dynamic() ? b_image_params_->batch_size : pshape[i].get_length();
            }

            io_dims_map_[output.get_any_name()] = dims;

            std::ostringstream engine_info;

            for (int j = 0; j < dims.nbDims; j++) {
                engine_info << ", ";
                engine_info << dims.d[j];
            }

            LOG_INFO(Logger::GetInstance(), "Output: {}, shape: {}", output.get_any_name(), engine_info.str());
        }
    }

    bool OpenVINOEngine::set_dynamic_shapes() {
        if (!model_) {
            LOG_ERROR(Logger::GetInstance(), "Model not loaded");
            return false;
        }

        LOG_INFO(Logger::GetInstance(), "Setting dynamic batch : ", b_image_params_->batch_size);

        for (auto& input : model_->inputs()) {
            auto shape = input.get_partial_shape();

            if (shape.rank().is_static() && shape.rank().get_length() >= 1) {
                shape[0] = b_image_params_->batch_size;
                model_->reshape({{input.get_any_name(), shape}});

                LOG_INFO(Logger::GetInstance(), "Input {} set to dynamic shape: {}", input.get_any_name(),
                    shape.to_string());
            }
        }

        model_->validate_nodes_and_infer_types();
        return true;
    }

    const std::unordered_map<std::string, ioDims>& OpenVINOEngine::get_io_dims() const {
        return io_dims_map_;
    }
} // namespace NexLab