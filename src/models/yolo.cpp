#include "models/yolo.hpp"

#include "models/post_process.hpp"
#include "models/pre_process.hpp"

namespace NexLab {
    void YOLO::pre_process_host(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) {
        if (images.size() != params_->batch_size) {
            LOG_ERROR(Logger::GetInstance(), "Input images size does not match batch size!");
            throw;
        }

        for (size_t input_idx = 0; input_idx < h_input_buffers_.size(); ++input_idx) {
            float* input_buffer = h_input_buffers_[input_idx];

            for (size_t batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
                cv::Mat resize_image;
                std::vector<float> blob_data;

                cv::Mat& image = images[batch_idx];
                ResizeParams& resize_param = img_resize_params[batch_idx];

                adaptive_resize(image, resize_image, params_->dst_w, params_->dst_h, resize_param);

                blob_from_image(resize_image, blob_data, params_->input_channels, params_->dst_w, params_->dst_h);

                size_t batch_offset = batch_idx * blob_data.size();
                std::memcpy(input_buffer + batch_offset, blob_data.data(), blob_data.size() * sizeof(float));
            }
        }
    }

    void YOLO::post_process_host(
        std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) {
        auto class_num = params_->num_class;
        auto detection_num = params_->num_detection;
        int key_point_num = params_->num_pose;
        int mask_num = params_->num_mask;
        int mask_size = params_->mask_size;
        int model_width = params_->dst_w;
        int model_height = params_->dst_h;

        if (outputs.size() != params_->batch_size) {
            LOG_ERROR(Logger::GetInstance(), "Output size does not match batch size");
            throw;
        }

        switch (params_->model_type) {
        case ModelType::YOLO_DETECTION:
            {
                outputs = decode_yolo_detection_host(h_output_buffers_, params_->batch_size, class_num, detection_num,
                    img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_POSE:
            {
                outputs = decode_yolo_pose_host(h_output_buffers_, params_->batch_size, class_num, key_point_num,
                    detection_num, img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_SEGMENT:
            {
                outputs =
                    decode_yolo_segment_host(h_output_buffers_, params_->batch_size, class_num, mask_num, detection_num,
                        mask_size, model_width, model_height, img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_OBB:
            {
                outputs = decode_yolo_obb_host(h_output_buffers_, params_->batch_size, class_num, detection_num,
                    img_resize_params, params_->confidence_threshold);
                break;
            }

        default:
            break;
        }

        for (size_t batch_idx = 0; batch_idx < params_->batch_size; ++batch_idx) {

            auto src_w = img_resize_params[batch_idx].image_width;
            auto src_h = img_resize_params[batch_idx].image_height;

            if (params_->model_type == ModelType::YOLO_OBB) {
                outputs[batch_idx] =
                    probiou_non_max_suppression(outputs[batch_idx], params_->iou_threshold, src_w, src_h);
            } else {
                outputs[batch_idx] = non_max_suppression(outputs[batch_idx], params_->iou_threshold, src_w, src_h);
            }
        }

        if (!outputs[0].empty()) {
            LOG_INFO(Logger::GetInstance(), "Detection class id: {}, box: [{} {} {} {}].",
                outputs[0][0].detection->class_id, outputs[0][0].detection->left, outputs[0][0].detection->top,
                outputs[0][0].detection->width, outputs[0][0].detection->height);
        }
    }

    void YOLO::pre_process_device(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) {
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
                compute_resize_params(img, img_resize_params[i], params_->dst_w, params_->dst_h);

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

            launch_yolo_preprocess_kernel(
                d_input, d_offsets, d_params, input_buffers_[idx], params_->batch_size, params_->dst_w, params_->dst_h);

            cudaFree(d_input);
            cudaFree(d_offsets);
            cudaFree(d_params);
        }
    }

    void YOLO::post_process_device(
        std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) {
        auto class_num = params_->num_class;
        auto detection_num = params_->num_detection;
        int key_point_num = params_->num_pose;
        int mask_num = params_->num_mask;
        int mask_size = params_->mask_size;
        int model_width = params_->dst_w;
        int model_height = params_->dst_h;

        if (outputs.size() != params_->batch_size) {
            LOG_ERROR(Logger::GetInstance(), "Output size does not match batch size");
            throw;
        }

        switch (params_->model_type) {
        case ModelType::YOLO_DETECTION:
            {
                outputs = decode_yolo_detection_device(output_buffers_, params_->batch_size, class_num, detection_num,
                    img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_POSE:
            {
                outputs = decode_yolo_pose_device(output_buffers_, params_->batch_size, class_num, key_point_num,
                    detection_num, img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_SEGMENT:
            {
                outputs =
                    decode_yolo_segment_device(output_buffers_, params_->batch_size, class_num, mask_num, detection_num,
                        mask_size, model_width, model_height, img_resize_params, params_->confidence_threshold);
                break;
            }

        case ModelType::YOLO_OBB:
            {
                outputs = decode_yolo_obb_device(output_buffers_, params_->batch_size, class_num, detection_num,
                    img_resize_params, params_->confidence_threshold);
                break;
            }

        default:
            break;
        }

        for (size_t batch_idx = 0; batch_idx < params_->batch_size; ++batch_idx) {

            auto src_w = img_resize_params[batch_idx].image_width;
            auto src_h = img_resize_params[batch_idx].image_height;

            if (params_->model_type == ModelType::YOLO_OBB) {
                outputs[batch_idx] =
                    probiou_non_max_suppression(outputs[batch_idx], params_->iou_threshold, src_w, src_h);
            } else {
                outputs[batch_idx] = non_max_suppression(outputs[batch_idx], params_->iou_threshold, src_w, src_h);
            }
        }

        if (params_->model_type == ModelType::YOLO_SEGMENT) {
            mask_decode_device(
                output_buffers_[1], outputs, mask_num, mask_size, model_width, model_height, img_resize_params);
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