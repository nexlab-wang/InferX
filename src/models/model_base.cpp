#include "models/model_base.hpp"

namespace NexLab {
    ModelBase::ModelBase(std::shared_ptr<BaseEngine> engine, std::shared_ptr<ImageModelParams> params)
        : engine_(std::move(engine)), params_(std::move(params)) {
        if (engine_) {
            engine_->load_dev();
        }

        engine_->initialize(*params_);
    }

    ModelBase::~ModelBase() {

        auto dev_type = engine_->dev_info_.get_dev_type();

        if (dev_type == DeviceType::DEV_GPU || dev_type == DeviceType::DEV_GPU_CPU) {
            for (auto& input_buffer : input_buffers_) {
                if (input_buffer != nullptr) {
                    cudaFree(input_buffer);
                }
            }

            for (auto& output_buffer : output_buffers_) {
                if (output_buffer != nullptr) {
                    cudaFree(output_buffer);
                }
            }
        } else if (dev_type == DeviceType::DEV_CPU) {

            for (auto& input_buffer : input_buffers_) {
                if (input_buffer != nullptr) {
                    free(input_buffer);
                    input_buffer = nullptr;
                }
            }

            for (auto& output_buffer : output_buffers_) {
                if (output_buffer != nullptr) {
                    free(output_buffer);
                    output_buffer = nullptr;
                }
            }
        }

        for (auto& h_input_buffer : h_input_buffers_) {
            if (h_input_buffer != nullptr) {
                free(h_input_buffer);
                h_input_buffer = nullptr;
            }
        }

        for (auto& h_output_buffer : h_output_buffers_) {
            if (h_output_buffer != nullptr) {
                free(h_output_buffer);
                h_output_buffer = nullptr;
            }
        }
    }

    bool ModelBase::load_model(const std::string& model_file) {
        if (!engine_->load_model(model_file)) {
            return false;
        }

        input_sizes_.clear();
        output_sizes_.clear();
        h_input_buffers_.clear();
        h_output_buffers_.clear();
        input_buffers_.clear();
        output_buffers_.clear();

        auto dev_type = engine_->dev_info_.get_dev_type();
        auto io_dims = engine_->get_io_dims();

        for (const auto& input : params_->model_input_names) {
            auto it = io_dims.find(input);

            if (it != io_dims.end()) {
                auto input_dims = it->second;
                size_t input_size = 1;

                if (input_dims.nbDims != -1) {
                    for (int i = 0; i < input_dims.nbDims; ++i) {
                        input_size *= input_dims.d[i];
                    }
                } else {
                    LOG_ERROR(Logger::GetInstance(), "Get input: {} size is failed.", input);
                    return false;
                }

                input_size *= sizeof(float);
                input_sizes_.emplace_back(input_size);
            }
        }

        for (const auto& output : params_->model_output_names) {
            auto it = io_dims.find(output);

            if (it != io_dims.end()) {
                auto output_dims = it->second;
                size_t output_size = 1;

                if (output_dims.nbDims != -1) {
                    for (int i = 0; i < output_dims.nbDims; ++i) {
                        output_size *= output_dims.d[i];
                    }
                } else {
                    LOG_ERROR(Logger::GetInstance(), "Get output: {} size is failed.", output);
                    return false;
                }

                output_size *= sizeof(float);
                output_sizes_.emplace_back(output_size);
            }
        }

        for (const auto& input_size : input_sizes_) {
            auto host_input_buffer = (float*) malloc(input_size);
            h_input_buffers_.emplace_back(host_input_buffer);

            if (dev_type == DeviceType::DEV_GPU || dev_type == DeviceType::DEV_GPU_CPU) {

                void* device_input_buffer;
                cudaMalloc(&device_input_buffer, input_size);
                input_buffers_.emplace_back(static_cast<float*>(device_input_buffer));

            } else if (dev_type == DeviceType::DEV_CPU) {

                float* input_buffer = (float*) malloc(input_size);
                input_buffers_.emplace_back(static_cast<float*>(input_buffer));
            }
        }

        for (const auto& output_size : output_sizes_) {
            auto host_output_buffer = (float*) malloc(output_size);
            h_output_buffers_.emplace_back(host_output_buffer);

            if (dev_type == DeviceType::DEV_GPU || dev_type == DeviceType::DEV_GPU_CPU) {

                void* device_output_buffer;
                cudaMalloc(&device_output_buffer, output_size);
                output_buffers_.emplace_back(static_cast<float*>(device_output_buffer));

            } else if (dev_type == DeviceType::DEV_CPU) {
                float* output_buffer = (float*) malloc(output_size);
                output_buffers_.emplace_back(static_cast<float*>(output_buffer));
            }
        }

        engine_->set_input_datas(input_buffers_);
        engine_->set_output_datas(output_buffers_);

        return true;
    }

    std::vector<std::vector<InferRes>> ModelBase::detection(std::vector<cv::Mat>& images) {

        std::vector<ResizeParams> image_resize_params(images.size());
        std::vector<std::vector<InferRes>> batch_result(images.size());

        auto dev_type = engine_->dev_info_.get_dev_type();

        if (dev_type == DeviceType::DEV_GPU) {

            pre_process_device(images, image_resize_params);

            engine_->infer_model();

            post_process_device(batch_result, image_resize_params);

        } else if (dev_type == DeviceType::DEV_CPU) {

            pre_process_host(images, image_resize_params);

            for (int i = 0; i < input_buffers_.size(); ++i) {
                std::memcpy(input_buffers_[i], h_input_buffers_[i], input_sizes_[i]);
            }

            engine_->infer_model();

            for (int i = 0; i < output_buffers_.size(); ++i) {
                std::memcpy(h_output_buffers_[i], output_buffers_[i], output_sizes_[i]);
            }

            post_process_host(batch_result, image_resize_params);

        } else if (dev_type == DeviceType::DEV_GPU_CPU) {

            pre_process_host(images, image_resize_params);

            for (int i = 0; i < input_buffers_.size(); ++i) {
                cudaMemcpy(input_buffers_[i], h_input_buffers_[i], input_sizes_[i], cudaMemcpyHostToDevice);
            }

            engine_->infer_model();

            for (int i = 0; i < output_buffers_.size(); ++i) {
                cudaMemcpy(h_output_buffers_[i], output_buffers_[i], output_sizes_[i], cudaMemcpyDeviceToHost);
            }

            post_process_host(batch_result, image_resize_params);
        }

        return batch_result;
    }
} // namespace NexLab