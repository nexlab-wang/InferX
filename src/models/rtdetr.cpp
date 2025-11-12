#include "models/rtdetr.hpp"

#include "models/post_process.hpp"
#include "models/pre_process.hpp"

namespace NexLab {
    void RTDETR::pre_process_host(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) {
        // resize+BGR->RGB+HWC->CHW+norm
        // transformer对于图像变形不敏感，可以直接进行resize
        if (images.size() != params_->batch_size) {
            throw std::runtime_error("Input images size does not match batch size!");
        }

        for (size_t input_idx = 0; input_idx < h_input_buffers_.size(); ++input_idx) {
            float* input_buffer = h_input_buffers_[input_idx];

            for (size_t batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
                std::vector<float> blob_data;
                cv::Mat& image = images[batch_idx];
                ResizeParams& resize_param = img_resize_params[batch_idx];

                auto model_width = params_->dst_w;
                auto model_height = params_->dst_h;

                resize_param.image_width = image.cols;
                resize_param.image_height = image.rows;

                cv::Mat resized_img(model_height, model_width, CV_8UC3);
                cv::resize(image, resized_img, resized_img.size(), 0, 0, cv::INTER_LINEAR);

                blob_from_image(resized_img, blob_data, params_->input_channels, params_->dst_w, params_->dst_h);

                size_t batch_offset = batch_idx * blob_data.size();
                std::memcpy(input_buffer + batch_offset, blob_data.data(), blob_data.size() * sizeof(float));
            }
        }
    }

    void RTDETR::post_process_host(
        std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) {
        auto model_type = params_->model_type;
        auto class_num = params_->num_class;
        auto detection_num = params_->num_detection;
        int key_point_num = params_->num_pose;
        int mask_num = params_->num_mask;
        int mask_size = params_->mask_size;
        int model_width = params_->dst_w;
        int model_height = params_->dst_h;

        if (model_type == ModelType::RT_DETR) {
            outputs = decode_rtdetr_detection_host(h_output_buffers_, params_->batch_size, class_num, detection_num,
                img_resize_params, params_->confidence_threshold);
        }

        if (!outputs[0].empty()) {
            for (const auto& output : outputs) {
                for (const auto& det : output) {
                    LOG_INFO(Logger::GetInstance(), "Detection class id: {}, box: [{} {} {} {}].",
                        det.detection->class_id, det.detection->left, det.detection->top, det.detection->width,
                        det.detection->height);
                }
            }
        }
    }

    void RTDETR::pre_process_device(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) {

        for (size_t idx = 0; idx < input_buffers_.size(); ++idx) {

            if (images.size() != params_->batch_size) {
                LOG_ERROR(Logger::GetInstance(), "Input images size does not match batch size!");
                throw;
            }

            std::vector<size_t> host_image_offsets(params_->batch_size);
            std::vector<uint8_t> host_pixel_data;
            size_t current_offset = 0;

            for (int i = 0; i < images.size(); ++i) {
                cv::Mat& img = images[i];

                ResizeParams& resize_param = img_resize_params[i];

                resize_param.image_width = img.cols;
                resize_param.image_height = img.rows;

                host_image_offsets[i] = current_offset;
                current_offset += img.total() * img.elemSize(); // w*h*channel

                // Image flatten
                cv::Mat contiguous_img = img.isContinuous() ? img : img.clone();
                host_pixel_data.insert(host_pixel_data.end(), contiguous_img.datastart, contiguous_img.dataend);
            }

            uint8_t* d_input = nullptr;
            size_t* d_offsets = nullptr;
            ResizeParams* d_params = nullptr;

            size_t input_size = host_pixel_data.size();

            cudaMalloc(&d_input, input_size);
            cudaMalloc(&d_offsets, params_->batch_size * sizeof(size_t));
            cudaMalloc(&d_params, params_->batch_size * sizeof(ResizeParams));

            cudaMemcpy(d_input, host_pixel_data.data(), input_size, cudaMemcpyHostToDevice);
            cudaMemcpy(
                d_offsets, host_image_offsets.data(), params_->batch_size * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(
                d_params, img_resize_params.data(), params_->batch_size * sizeof(ResizeParams), cudaMemcpyHostToDevice);

            launch_rtdetr_preprocess_kernel(
                d_input, d_offsets, input_buffers_[idx], params_->batch_size, params_->dst_w, params_->dst_h, d_params);

            cudaFree(d_input);
            cudaFree(d_offsets);
            cudaFree(d_params);
        }
    }

    void RTDETR::post_process_device(
        std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) {
        if (outputs.size() != params_->batch_size) {
            LOG_ERROR(Logger::GetInstance(), "Output size does not match batch size.");
            throw;
        }

        if (params_->model_type == ModelType::RT_DETR) {
            auto model_params = params_.get();
            outputs = decode_rtdetr_detection_device(output_buffers_, model_params, img_resize_params);
        }

        if (!outputs[0].empty()) {
            for (const auto& output : outputs) {
                for (const auto& det : output) {
                    LOG_INFO(Logger::GetInstance(), "Detection class id: {}, box: [{} {} {} {}].",
                        det.detection->class_id, det.detection->left, det.detection->top, det.detection->width,
                        det.detection->height);
                }
            }
        }
    }
} // namespace NexLab