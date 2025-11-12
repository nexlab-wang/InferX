#ifndef __POST_PROCESS__
#define __POST_PROCESS__
#include "common/logger.hpp"
#include "kernel_function/kernel_function.hpp"
#include "models/model_utils.hpp"
#include "opencv2/opencv.hpp"

namespace NexLab {

    std::vector<std::vector<InferRes>> inline decode_yolo_detection_host(const std::vector<float*>& h_output_buffers,
        int batch_size, int class_num, int detect_num, const std::vector<ResizeParams>& img_resize_params,
        float conf_th) {
        std::vector<std::vector<InferRes>> batch_results(batch_size);

        if (h_output_buffers.empty()) {
            std::cerr << "Host output buffers is empty." << std::endl;
            return {};
        }

        float* output_data = h_output_buffers.front();
        int feat_num = 4 + class_num;

        for (int i = 0; i < batch_size; ++i) {
            std::vector<InferRes> detect_results;

            for (int j = 0; j < detect_num; ++j) {
                Detection detection;
                InferRes result;

                int class_id = -1;
                float max_class_score = -1.0f;

                int batch_offset = i * feat_num * detect_num;

                decode_class(output_data, batch_offset, j, detect_num, class_num, class_id, max_class_score);

                if (max_class_score < conf_th)
                    continue;

                decode_bbox(output_data, batch_offset, j, detect_num, img_resize_params[i], detection);

                detection.class_id = class_id;
                detection.confidence = max_class_score;

                result.detection = detection;
                detect_results.emplace_back(result);
            }
            batch_results[i] = detect_results;
        }

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_pose_host(const std::vector<float*>& h_output_buffers,
        int batch_size, int class_num, int key_point_num, int detect_num,
        const std::vector<ResizeParams>& img_resize_params, float conf_th) {
        std::vector<std::vector<InferRes>> batch_results(batch_size);

        if (h_output_buffers.empty()) {
            std::cerr << "Host output buffers is empty." << std::endl;
            return {};
        }

        float* output_data = h_output_buffers.front();
        int feat_num = 4 + class_num + 3 * key_point_num;


        for (int i = 0; i < batch_size; ++i) {
            std::vector<InferRes> detect_results;
            for (int j = 0; j < detect_num; ++j) {

                Detection detection;
                Pose pose_points;
                InferRes result;

                int class_id = -1;
                float max_class_score = -1.0f;

                int batch_offset = i * feat_num * detect_num;

                decode_class(output_data, batch_offset, j, detect_num, class_num, class_id, max_class_score);

                if (max_class_score < conf_th)
                    continue;

                decode_bbox(output_data, batch_offset, j, detect_num, img_resize_params[i], detection);

                for (int k = 0; k < key_point_num; ++k) {

                    float point_x = output_data[i * feat_num * detect_num + (4 + class_num + k * 3) * detect_num + j];
                    float point_y =
                        output_data[i * feat_num * detect_num + (4 + class_num + 1 + k * 3) * detect_num + j];
                    float point_score =
                        output_data[i * feat_num * detect_num + (4 + class_num + 2 + k * 3) * detect_num + j];

                    float keypoint_x =
                        static_cast<float>((point_x - img_resize_params[i].pad_w) / img_resize_params[i].ratio);
                    float keypoint_y =
                        static_cast<float>((point_y - img_resize_params[i].pad_h) / img_resize_params[i].ratio);
                    float keypoint_score = point_score;

                    std::vector<float> key_point = {keypoint_x, keypoint_y, keypoint_score};
                    pose_points.keypoints.emplace_back(key_point);
                }

                detection.class_id = class_id;
                detection.confidence = max_class_score;

                result.detection = detection;
                result.pose = pose_points;
                detect_results.emplace_back(result);
            }
            batch_results[i] = detect_results;
        }

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_segment_host(const std::vector<float*>& h_output_buffers,
        int batch_size, int class_num, int mask_num, int detect_num, int mask_size, int model_width, int model_height,
        const std::vector<ResizeParams>& img_resize_params, float conf_th) {

        std::vector<std::vector<InferRes>> batch_results(batch_size);

        if (h_output_buffers.empty()) {
            std::cerr << "Host output buffers is empty." << std::endl;
            return {};
        }

        float* output_data = h_output_buffers[0];
        float* mask_data = h_output_buffers[1];
        int feat_num = 4 + class_num + mask_num;

        int mask_proto_size = mask_num * mask_size * mask_size;

        for (int i = 0; i < batch_size; ++i) {
            std::vector<InferRes> detect_results;

            float* current_mask_data = mask_data + i * mask_proto_size;

            for (int j = 0; j < detect_num; ++j) {
                Detection detection;
                InferRes result;

                int class_id = -1;
                float max_class_score = -1.0f;

                int batch_offset = i * feat_num * detect_num;

                decode_class(output_data, batch_offset, j, detect_num, class_num, class_id, max_class_score);

                if (max_class_score < conf_th)
                    continue;

                float cx = output_data[batch_offset + 0 * detect_num + j];
                float cy = output_data[batch_offset + 1 * detect_num + j];
                float cw = output_data[batch_offset + 2 * detect_num + j];
                float ch = output_data[batch_offset + 3 * detect_num + j];

                float ratio = img_resize_params[i].ratio;
                int pad_w = img_resize_params[i].pad_w;
                int pad_h = img_resize_params[i].pad_h;

                float x = (cx - pad_w) / ratio;
                float y = (cy - pad_h) / ratio;
                float w = cw / ratio;
                float h = ch / ratio;

                int left = std::max(static_cast<int>(x - 0.5 * w), 0);
                int top = std::max(static_cast<int>(y - 0.5 * h), 0);
                int width = static_cast<int>(w);
                int height = static_cast<int>(h);

                if (width <= 0 || height <= 0) {
                    continue;
                }

                /* Mask Decoding Process:
                    Step 1: Extract mask coefficients from output data
                    These coefficients will be used to combine prototype masks */
                std::vector<float> mask_coefficients;
                for (int k = 0; k < mask_num; ++k) {
                    float mask_coeff = output_data[batch_offset + (4 + class_num + k) * detect_num + j];
                    mask_coefficients.push_back(mask_coeff);
                }

                /* Step 2: Initialize empty mask and combine prototype masks
                    using the coefficients (linear combination) */
                cv::Mat mask = cv::Mat::zeros(mask_size, mask_size, CV_32F);
                for (int k = 0; k < mask_num; ++k) {
                    // Get current mask prototype from mask_data
                    cv::Mat mask_proto =
                        cv::Mat(mask_size, mask_size, CV_32F, current_mask_data + k * mask_size * mask_size);
                    // Weighted sum of prototype masks
                    mask += mask_coefficients[k] * mask_proto;
                }

                /* Step 3: Apply sigmoid activation to the combined mask
                    to get values between 0 and 1 */
                cv::exp(-mask, mask);
                mask = 1.0 / (1.0 + mask);

                /* Step 4: Resize the mask to model input dimensions
                    (typically 640x640 for YOLOv8 segmentation) */
                cv::Mat resized_mask;
                cv::resize(mask, resized_mask, cv::Size(model_width, model_height), 0, 0, cv::INTER_LINEAR);

                /* Step 5: Calculate the bounding box in model dimensions
                    where the mask will be cropped from(model_width*model_height) */
                int box_left = static_cast<int>(cx - 0.5 * cw);
                int box_top = static_cast<int>(cy - 0.5 * ch);
                int box_width = static_cast<int>(cw);
                int box_height = static_cast<int>(ch);

                // Clamp box coordinates to model dimensions
                box_left = std::max(0, box_left);
                box_top = std::max(0, box_top);
                box_width = std::min(box_width, model_width - box_left);
                box_height = std::min(box_height, model_height - box_top);

                /* Step 6: Crop the mask region corresponding to the detected object */
                cv::Rect box_rect(box_left, box_top, box_width, box_height);
                cv::Mat mask_box = resized_mask(box_rect).clone();

                /* Step 7: Resize the cropped mask to match the original image dimensions
                    of the detected object */
                cv::Mat resized_mask_box;
                cv::resize(mask_box, resized_mask_box, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

                detection.class_id = class_id;
                detection.confidence = max_class_score;
                detection.left = left;
                detection.top = top;
                detection.width = width;
                detection.height = height;

                result.detection = detection;
                result.mask_coefficient = mask_coefficients;
                result.segment_mask = resized_mask_box;

                detect_results.emplace_back(result);
            }

            batch_results[i] = detect_results;
        }

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_obb_host(const std::vector<float*>& h_output_buffers,
        int batch_size, int class_num, int detect_num, const std::vector<ResizeParams>& img_resize_params,
        float conf_th) {
        std::vector<std::vector<InferRes>> batch_results(batch_size);

        if (h_output_buffers.empty()) {
            std::cerr << "Host output buffers is empty." << std::endl;
            return {};
        }

        float* output_data = h_output_buffers.front(); // xywh cls angle
        int feat_num = 5 + class_num;

        for (int i = 0; i < batch_size; ++i) {
            std::vector<InferRes> detect_results;
            for (int j = 0; j < detect_num; ++j) {
                Detection detection;
                InferRes result;

                int class_id = -1;
                float max_class_score = -1.0f;

                int batch_offset = i * feat_num * detect_num;

                decode_class(output_data, batch_offset, j, detect_num, class_num, class_id, max_class_score);

                if (max_class_score < conf_th)
                    continue;

                decode_bbox(output_data, batch_offset, j, detect_num, img_resize_params[i], detection);

                float angle = output_data[batch_offset + (feat_num - 1) * detect_num + j];

                detection.class_id = class_id;
                detection.confidence = max_class_score;
                detection.angle = angle;

                result.detection = detection;
                detect_results.emplace_back(result);
            }
            batch_results[i] = detect_results;
        }

        return batch_results;
    }


    std::vector<std::vector<InferRes>> inline decode_yolo_detection_device(const std::vector<float*>& output_buffers,
        int batch_size, int class_num, int detect_num, const std::vector<ResizeParams>& img_resize_params,
        float conf_th) {
        if (output_buffers.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Device output buffers is empty.");
            return {};
        }

        int* d_valid_counts;
        int* d_valid_indices;
        Detection* d_detections;
        ResizeParams* d_resize_params;

        size_t valid_counts_size = batch_size * sizeof(int);
        size_t valid_indices_size = batch_size * detect_num * sizeof(int);
        size_t detections_size = batch_size * detect_num * sizeof(Detection);
        size_t resize_params_size = batch_size * sizeof(ResizeParams);

        cudaMalloc(&d_valid_counts, valid_counts_size);
        cudaMalloc(&d_valid_indices, valid_indices_size);
        cudaMalloc(&d_detections, detections_size);
        cudaMalloc(&d_resize_params, resize_params_size);

        cudaMemset(d_valid_counts, 0, valid_counts_size);
        cudaMemset(d_valid_indices, -1, valid_indices_size);
        cudaMemcpy(d_resize_params, img_resize_params.data(), resize_params_size, cudaMemcpyHostToDevice);

        launch_decode_detections_kernel(output_buffers[0], d_resize_params, batch_size, class_num, detect_num, conf_th,
            d_valid_counts, d_valid_indices, d_detections);

        std::vector<int> h_valid_counts(batch_size);
        std::vector<int> h_valid_indices(batch_size * detect_num);
        std::vector<Detection> h_detections(batch_size * detect_num);

        cudaMemcpy(h_valid_counts.data(), d_valid_counts, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_valid_indices.data(), d_valid_indices, batch_size * detect_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_detections.data(), d_detections, batch_size * detect_num * sizeof(Detection), cudaMemcpyDeviceToHost);

        std::vector<std::vector<InferRes>> batch_results(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            int count = h_valid_counts[i];
            int base_idx = i * detect_num;

            for (int k = 0; k < count; ++k) {
                int full_idx = h_valid_indices[base_idx + k];
                const Detection& det = h_detections[full_idx];

                InferRes res;
                res.detection = det;

                batch_results[i].emplace_back(res);
            }
        }

        cudaFree(d_valid_counts);
        cudaFree(d_valid_indices);
        cudaFree(d_detections);
        cudaFree(d_resize_params);

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_pose_device(const std::vector<float*>& output_buffers,
        int batch_size, int class_num, int key_point_num, int detect_num,
        const std::vector<ResizeParams>& img_resize_params, float conf_th) {
        if (output_buffers.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Device output buffers is empty.");
            return {};
        }

        int* d_valid_counts; // 每个batch的有效值计数
        int* d_valid_indices; // 所有有效值索引
        float* d_key_points_flat; // keypoint数组
        Detection* d_detections;
        ResizeParams* d_resize_params;

        size_t valid_counts_size = batch_size * sizeof(int);
        size_t valid_indices_size = batch_size * detect_num * sizeof(int);
        size_t key_points_size = batch_size * detect_num * key_point_num * 3 * sizeof(float);
        size_t detections_size = batch_size * detect_num * sizeof(Detection);
        size_t resize_params_size = batch_size * sizeof(ResizeParams);

        cudaMalloc(&d_valid_counts, valid_counts_size);
        cudaMalloc(&d_valid_indices, valid_indices_size);
        cudaMalloc(&d_key_points_flat, key_points_size);
        cudaMalloc(&d_detections, detections_size);
        cudaMalloc(&d_resize_params, resize_params_size);

        cudaMemset(d_valid_counts, 0, valid_counts_size);
        cudaMemset(d_valid_indices, -1, valid_indices_size);
        cudaMemcpy(d_resize_params, img_resize_params.data(), resize_params_size, cudaMemcpyHostToDevice);

        launch_decode_pose_kernel(output_buffers[0], d_resize_params, batch_size, class_num, key_point_num, detect_num,
            conf_th, d_valid_counts, d_valid_indices, d_detections, d_key_points_flat);


        std::vector<int> h_valid_counts(batch_size);
        std::vector<int> h_valid_indices(batch_size * detect_num);
        std::vector<Detection> h_detections(batch_size * detect_num);
        std::vector<float> h_key_points_flat(batch_size * detect_num * key_point_num * 3);

        cudaMemcpy(h_valid_counts.data(), d_valid_counts, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_valid_indices.data(), d_valid_indices, batch_size * detect_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_key_points_flat.data(), d_key_points_flat,
            batch_size * detect_num * key_point_num * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_detections.data(), d_detections, batch_size * detect_num * sizeof(Detection), cudaMemcpyDeviceToHost);

        std::vector<std::vector<InferRes>> batch_results(batch_size);
        for (int i = 0; i < batch_size; ++i) {

            int count = h_valid_counts[i];
            int base_idx = i * detect_num;

            for (int k = 0; k < count; ++k) {
                int full_idx = h_valid_indices[base_idx + k]; // 有效值在h_detections中的全局索引
                const Detection& det = h_detections[full_idx];

                Pose key_points;
                float* kp_src = h_key_points_flat.data() + full_idx * key_point_num * 3;

                for (int p = 0; p < key_point_num; p++) {
                    std::vector<float> key_point = {kp_src[p * 3], kp_src[p * 3 + 1], kp_src[p * 3 + 2]};
                    key_points.keypoints.emplace_back(key_point);
                }

                InferRes res;
                res.detection = det;
                res.pose = key_points;
                batch_results[i].emplace_back(res);
            }
        }

        cudaFree(d_valid_counts);
        cudaFree(d_valid_indices);
        cudaFree(d_key_points_flat);
        cudaFree(d_detections);
        cudaFree(d_resize_params);

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_segment_device(const std::vector<float*>& output_buffers,
        int batch_size, int class_num, int mask_num, int detect_num, int mask_size, int model_width, int model_height,
        const std::vector<ResizeParams>& img_resize_params, float conf_th) {

        if (output_buffers.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Device output buffers is empty.");
            return {};
        }

        int* d_valid_counts;
        int* d_valid_indices;
        Detection* d_detections;
        float* d_mask_coeffs;
        ResizeParams* d_resize_params;

        // detect_num是yolo输出检测框总数
        cudaMalloc(&d_valid_counts, batch_size * sizeof(int));
        cudaMalloc(&d_valid_indices, batch_size * detect_num * sizeof(int));
        cudaMalloc(&d_resize_params, batch_size * sizeof(ResizeParams));
        cudaMalloc(&d_detections, batch_size * detect_num * sizeof(Detection));
        cudaMalloc(&d_mask_coeffs, batch_size * detect_num * mask_num * sizeof(float));

        cudaMemset(d_valid_counts, 0, batch_size * sizeof(int));
        cudaMemset(d_valid_indices, -1, batch_size * detect_num * sizeof(int));
        cudaMemcpy(
            d_resize_params, img_resize_params.data(), batch_size * sizeof(ResizeParams), cudaMemcpyHostToDevice);

        launch_decode_boxes_kernel(output_buffers[0], d_resize_params, batch_size, class_num, mask_num, detect_num,
            conf_th, d_valid_counts, d_valid_indices, d_detections, d_mask_coeffs);

        // 分配主机内存
        std::vector<int> h_valid_counts(batch_size);
        std::vector<int> h_valid_indices(batch_size * detect_num);
        std::vector<Detection> h_detections(batch_size * detect_num);
        std::vector<float> h_mask_coeffs(batch_size * detect_num * mask_num);

        cudaMemcpy(h_valid_counts.data(), d_valid_counts, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_valid_indices.data(), d_valid_indices, batch_size * detect_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_detections.data(), d_detections, batch_size * detect_num * sizeof(Detection), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mask_coeffs.data(), d_mask_coeffs, batch_size * detect_num * mask_num * sizeof(float),
            cudaMemcpyDeviceToHost);

        std::vector<std::vector<InferRes>> batch_results(batch_size);
        for (int i = 0; i < batch_size; ++i) {

            int count = h_valid_counts[i];
            int base_idx = i * detect_num;

            for (int k = 0; k < count; ++k) {
                int full_idx = h_valid_indices[base_idx + k];
                const Detection& det = h_detections[full_idx];

                int coeff_start = full_idx * mask_num;
                if (coeff_start + mask_num > h_mask_coeffs.size()) {
                    continue;
                }
                std::vector<float> mask_coeffs(
                    h_mask_coeffs.begin() + coeff_start, h_mask_coeffs.begin() + coeff_start + mask_num);

                InferRes res;
                res.detection = det;
                res.mask_coefficient = mask_coeffs;
                batch_results[i].emplace_back(res);
            }
        }

        cudaFree(d_valid_counts);
        cudaFree(d_valid_indices);
        cudaFree(d_detections);
        cudaFree(d_mask_coeffs);
        cudaFree(d_resize_params);

        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_yolo_obb_device(const std::vector<float*>& output_buffers,
        int batch_size, int class_num, int detect_num, const std::vector<ResizeParams>& img_resize_params,
        float conf_th) {
        if (output_buffers.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Device output buffers is empty.");
            return {};
        }

        int* d_valid_counts;
        int* d_valid_indices;
        Detection* d_detections;
        ResizeParams* d_resize_params;

        size_t valid_counts_size = batch_size * sizeof(int);
        size_t valid_indices_size = batch_size * detect_num * sizeof(int);
        size_t detections_size = batch_size * detect_num * sizeof(Detection);
        size_t resize_params_size = batch_size * sizeof(ResizeParams);

        cudaMalloc(&d_valid_counts, valid_counts_size);
        cudaMalloc(&d_valid_indices, valid_indices_size);
        cudaMalloc(&d_detections, detections_size);
        cudaMalloc(&d_resize_params, resize_params_size);

        cudaMemset(d_valid_counts, 0, valid_counts_size);
        cudaMemset(d_valid_indices, -1, valid_indices_size);
        cudaMemcpy(d_resize_params, img_resize_params.data(), resize_params_size, cudaMemcpyHostToDevice);


        launch_decode_obb_kernel(output_buffers[0], d_resize_params, batch_size, class_num, detect_num, conf_th,
            d_valid_counts, d_valid_indices, d_detections);

        std::vector<int> h_valid_counts(batch_size);
        std::vector<int> h_valid_indices(batch_size * detect_num);
        std::vector<Detection> h_detections(batch_size * detect_num);

        cudaMemcpy(h_valid_counts.data(), d_valid_counts, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_valid_indices.data(), d_valid_indices, batch_size * detect_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_detections.data(), d_detections, batch_size * detect_num * sizeof(Detection), cudaMemcpyDeviceToHost);

        std::vector<std::vector<InferRes>> batch_results(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            int count = h_valid_counts[i];
            int base_idx = i * detect_num;

            for (int k = 0; k < count; ++k) {
                int full_idx = h_valid_indices[base_idx + k];
                const Detection& det = h_detections[full_idx];

                InferRes res;
                res.detection = det;
                batch_results[i].emplace_back(res);
            }
        }

        cudaFree(d_valid_counts);
        cudaFree(d_valid_indices);
        cudaFree(d_detections);
        cudaFree(d_resize_params);

        return batch_results;
    }

    void inline mask_decode_device(float* output_mask, std::vector<std::vector<InferRes>>& nms_detections, int mask_num,
        int mask_size, int model_width, int model_height, const std::vector<ResizeParams>& img_resize_params) {
        if (output_mask == nullptr) {
            LOG_ERROR(Logger::GetInstance(), "Mask output is empty");
            return;
        }

        int total_detections = 0;
        std::vector<int> h_batch_indices;

        for (int batch_idx = 0; batch_idx < nms_detections.size(); ++batch_idx) {
            for (int i = 0; i < nms_detections[batch_idx].size(); ++i) {
                h_batch_indices.push_back(batch_idx);
            }
            total_detections += nms_detections[batch_idx].size();
        }

        if (total_detections == 0) {
            return;
        }

        float* d_mask_coeffs = nullptr;
        float* d_resize_mask = nullptr;
        int* d_batch_indices = nullptr;

        size_t mask_coeffs_size = total_detections * mask_num * sizeof(float);
        size_t resize_mask_size = total_detections * model_height * model_width * sizeof(float);
        size_t batch_indices_size = total_detections * sizeof(int);

        if (cudaMalloc(&d_mask_coeffs, mask_coeffs_size) != cudaSuccess
            || cudaMalloc(&d_resize_mask, resize_mask_size) != cudaSuccess
            || cudaMalloc(&d_batch_indices, batch_indices_size) != cudaSuccess) {
            LOG_ERROR(Logger::GetInstance(), "CUDA memory allocation failed");
            if (d_mask_coeffs)
                cudaFree(d_mask_coeffs);
            if (d_resize_mask)
                cudaFree(d_resize_mask);
            if (d_batch_indices)
                cudaFree(d_batch_indices);
            return;
        }

        std::vector<float> h_mask_coeffs(total_detections * mask_num);
        int det_idx = 0;
        for (const auto& batch_det : nms_detections) {
            for (const auto& det : batch_det) {
                auto coeffs = det.mask_coefficient.value();
                if (coeffs.size() != static_cast<size_t>(mask_num)) {
                    LOG_ERROR(Logger::GetInstance(), "Mask coefficients size mismatch");
                    cudaFree(d_mask_coeffs);
                    cudaFree(d_resize_mask);
                    return;
                }
                std::copy(coeffs.begin(), coeffs.end(), h_mask_coeffs.begin() + det_idx * mask_num);
                det_idx++;
            }
        }


        if (cudaMemcpy(
                d_mask_coeffs, h_mask_coeffs.data(), h_mask_coeffs.size() * sizeof(float), cudaMemcpyHostToDevice)
                != cudaSuccess
            || cudaMemcpy(d_batch_indices, h_batch_indices.data(), batch_indices_size, cudaMemcpyHostToDevice)
                   != cudaSuccess) {
            LOG_ERROR(Logger::GetInstance(), "Coefficients copy failed");
            cudaFree(d_mask_coeffs);
            cudaFree(d_resize_mask);
            return;
        }

        launch_decode_mask_batch_kernel(output_mask, d_mask_coeffs, d_resize_mask, d_batch_indices,
            nms_detections.size(), total_detections, mask_num, mask_size, model_width, model_height);

        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            LOG_ERROR(Logger::GetInstance(), "Kernel launch failed: {}", cudaGetErrorString(kernelErr));
            cudaFree(d_mask_coeffs);
            cudaFree(d_resize_mask);
            cudaFree(d_batch_indices);
            return;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR(Logger::GetInstance(), "CUDA error during synchronization: {}", cudaGetErrorString(err));
            cudaFree(d_mask_coeffs);
            cudaFree(d_resize_mask);
            cudaFree(d_batch_indices);
            return;
        }

        std::vector<float> h_resize_mask(total_detections * model_width * model_height);
        if (cudaMemcpy(h_resize_mask.data(), d_resize_mask, resize_mask_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG_ERROR(Logger::GetInstance(), "Result copy failed");
        }

        int mask_offset = 0;
        for (size_t batch_idx = 0; batch_idx < nms_detections.size(); ++batch_idx) {

            for (size_t det_idx = 0; det_idx < nms_detections[batch_idx].size(); ++det_idx) {
                cv::Mat model_mask = cv::Mat(
                    model_height, model_width, CV_32F, h_resize_mask.data() + mask_offset * model_width * model_height);
                mask_offset++;
                int box_left = static_cast<int>(nms_detections[batch_idx][det_idx].detection.value().left);
                int box_top = static_cast<int>(nms_detections[batch_idx][det_idx].detection.value().top);
                int box_width = static_cast<int>(nms_detections[batch_idx][det_idx].detection.value().width);
                int box_height = static_cast<int>(nms_detections[batch_idx][det_idx].detection.value().height);

                // 获取放缩参数
                float ratio = img_resize_params[batch_idx].ratio;
                int pad_w = img_resize_params[batch_idx].pad_w;
                int pad_h = img_resize_params[batch_idx].pad_h;

                float cx = (box_left + 0.5 * box_width) * ratio + pad_w; // 模型输入坐标系中的中心点 x
                float cy = (box_top + 0.5 * box_height) * ratio + pad_h; // 模型输入坐标系中的中心点 y
                float cw = box_width * ratio; // 模型输入坐标系中的宽度
                float ch = box_height * ratio; // 模型输入坐标系中的高度

                int model_box_left = static_cast<int>(cx - 0.5 * cw);
                int model_box_top = static_cast<int>(cy - 0.5 * ch);
                int model_box_width = static_cast<int>(cw);
                int model_box_height = static_cast<int>(ch);

                model_box_left = std::max(0, model_box_left);
                model_box_top = std::max(0, model_box_top);
                model_box_width = std::min(model_box_width, model_width - model_box_left);
                model_box_height = std::min(model_box_height, model_height - model_box_top);

                cv::Rect model_box_rect(model_box_left, model_box_top, model_box_width, model_box_height);
                cv::Mat mask_box = model_mask(model_box_rect).clone();

                cv::Mat resized_mask_box;
                cv::resize(mask_box, resized_mask_box, cv::Size(box_width, box_height), 0, 0, cv::INTER_LINEAR);

                nms_detections[batch_idx][det_idx].segment_mask = resized_mask_box;
            }
        }

        cudaFree(d_mask_coeffs);
        cudaFree(d_resize_mask);
        cudaFree(d_batch_indices);
    }

    std::vector<InferRes> inline non_max_suppression(
        const std::vector<InferRes>& detect_res, float iou_threshold, int img_width, int img_height) {
        auto detect_result = detect_res; // const对象无法进行sort
        std::vector<InferRes> nms_res;

        // 首先按照类别进行分组
        std::unordered_map<int, std::vector<int>> class_to_indices;
        for (int i = 0; i < detect_result.size(); ++i) {
            int class_id = detect_result[i].detection->class_id;
            class_to_indices[class_id].push_back(i);
        }

        // 对每个类别分别进行NMS
        for (const auto& class_indices : class_to_indices) {
            std::list<int> indices(class_indices.second.begin(), class_indices.second.end());

            // 使用 std::list 的成员函数 sort 进行排序
            indices.sort([&detect_result](int a, int b) {
                return detect_result[a].detection->confidence > detect_result[b].detection->confidence;
            });

            while (!indices.empty()) {
                int best_idx = indices.front();
                float left = detect_result[best_idx].detection->left;
                float top = detect_result[best_idx].detection->top;
                float width = detect_result[best_idx].detection->width;
                float height = detect_result[best_idx].detection->height;
                std::vector<float> bbox = {left, top, width, height};

                if (!bbox.empty()) {
                    if (is_in_image(bbox, img_width, img_height)) {
                        nms_res.emplace_back(detect_result[best_idx]);
                    }
                }

                indices.pop_front();

                for (auto it = indices.begin(); it != indices.end();) {
                    float left = detect_result[*it].detection->left;
                    float top = detect_result[*it].detection->top;
                    float width = detect_result[*it].detection->width;
                    float height = detect_result[*it].detection->height;
                    std::vector<float> cur_bbox = {left, top, width, height};

                    if (compute_iou(bbox, cur_bbox) > iou_threshold) {
                        it = indices.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }

        return nms_res;
    }

    std::vector<InferRes> inline probiou_non_max_suppression(
        const std::vector<InferRes>& detect_res, float prob_iou_th, int img_width, int img_height) {
        if (detect_res.empty())
            return {};

        // 1. 按置信度降序排序
        std::vector<int> indices(detect_res.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int i, int j) { return detect_res[i].detection->confidence > detect_res[j].detection->confidence; });

        std::vector<bool> suppressed(detect_res.size(), false);
        std::vector<InferRes> final_detections;

        // 2. 进行 probIoU NMS
        for (size_t i = 0; i < indices.size(); ++i) {
            if (suppressed[indices[i]])
                continue;

            const InferRes& det1 = detect_res[indices[i]];
            final_detections.push_back(det1);

            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (suppressed[indices[j]])
                    continue;

                const InferRes& det2 = detect_res[indices[j]];
                float prob_iou = compute_prob_iou(det1, det2);

                if (prob_iou > prob_iou_th) {
                    suppressed[indices[j]] = true;
                }
            }
        }

        return final_detections;
    }

    std::vector<std::vector<InferRes>> inline decode_rtdetr_detection_host(const std::vector<float*>& h_output_buffers,
        int batch_size, int num_classes, int num_detections, const std::vector<ResizeParams>& img_resize_params,
        float conf_th) {

        // xywh2xyxy+阈值过滤+坐标映射

        if (h_output_buffers.empty()) {
            std::cerr << "Host output buffers is empty." << std::endl;
            return {};
        }

        float* output_data = h_output_buffers.front();
        int feat_num = 4 + num_classes;

        std::vector<std::vector<InferRes>> batch_results(batch_size);

        for (int i = 0; i < batch_size; ++i) {

            ResizeParams resize_param = img_resize_params[i];
            int img_width = resize_param.image_width;
            int img_height = resize_param.image_height;

            std::vector<InferRes> detect_results;
            for (int j = 0; j < num_detections; ++j) {
                float* detection = &output_data[i * feat_num * num_detections + j * feat_num];

                float cx = detection[0];
                float cy = detection[1];
                float cw = detection[2];
                float ch = detection[3];

                int class_id = -1;
                float max_class_score = -1.0f;
                for (int k = 0; k < num_classes; ++k) {
                    float class_score = detection[4 + k];
                    if (class_score > max_class_score) {
                        max_class_score = class_score;
                        class_id = k;
                    }
                }

                if (max_class_score > conf_th) {
                    int xmin = std::max(static_cast<int>((cx - 0.5 * cw) * img_width), 0);
                    int ymin = std::max(static_cast<int>((cy - 0.5 * ch) * img_height), 0);
                    int xmax = std::min(static_cast<int>((cx + 0.5 * cw) * img_width), img_width - 1);
                    int ymax = std::min(static_cast<int>((cy + 0.5 * ch) * img_height), img_height - 1);

                    int width = xmax - xmin;
                    int height = ymax - ymin;

                    if (width <= 0 || height <= 0) {
                        continue;
                    }

                    Detection detect;
                    detect.class_id = class_id;
                    detect.confidence = max_class_score;
                    detect.left = xmin;
                    detect.top = ymin;
                    detect.width = width;
                    detect.height = height;


                    InferRes result;
                    result.detection = detect;
                    detect_results.emplace_back(result);
                }
            }
            batch_results[i] = detect_results;
        }
        return batch_results;
    }

    std::vector<std::vector<InferRes>> inline decode_rtdetr_detection_device(const std::vector<float*>& output_buffers,
        const ImageModelParams* params, std::vector<ResizeParams>& img_resize_params) {

        if (output_buffers.empty()) {
            LOG_ERROR(Logger::GetInstance(), "Device output buffers is empty.");
            return {};
        }

        int batch_size = params->batch_size;
        int detection_num = params->num_detection;
        int class_num = params->num_class;
        float conf_th = params->confidence_threshold;

        int* d_valid_counts;
        int* d_valid_indices;
        Detection* d_detections;
        ResizeParams* d_resize_params;

        size_t valid_counts_size = batch_size * sizeof(int);
        size_t valid_indices_size = batch_size * detection_num * sizeof(int);
        size_t detections_size = batch_size * detection_num * sizeof(Detection);
        size_t resize_params_size = batch_size * sizeof(ResizeParams);

        cudaMalloc(&d_valid_counts, valid_counts_size);
        cudaMalloc(&d_valid_indices, valid_indices_size);
        cudaMalloc(&d_detections, detections_size);
        cudaMalloc(&d_resize_params, resize_params_size);

        cudaMemset(d_valid_counts, 0, valid_counts_size);
        cudaMemset(d_valid_indices, -1, valid_indices_size);
        cudaMemcpy(d_resize_params, img_resize_params.data(), resize_params_size, cudaMemcpyHostToDevice);

        launch_decode_rtdetr_kernel(output_buffers[0], d_resize_params, batch_size, detection_num, class_num, conf_th,
            d_valid_counts, d_valid_indices, d_detections);

        std::vector<int> h_valid_counts(batch_size);
        std::vector<int> h_valid_indices(batch_size * detection_num);
        std::vector<Detection> h_detections(batch_size * detection_num);

        cudaMemcpy(h_valid_counts.data(), d_valid_counts, valid_counts_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_valid_indices.data(), d_valid_indices, valid_indices_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_detections.data(), d_detections, detections_size, cudaMemcpyDeviceToHost);

        std::vector<std::vector<InferRes>> batch_results(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            int count = h_valid_counts[i]; // 每个batch有效值计数
            int base_idx = i * detection_num;

            for (int j = 0; j < count; ++j) {
                int full_idx = h_valid_indices[base_idx + j];
                const Detection& det = h_detections[full_idx];

                InferRes result;
                result.detection = det;

                batch_results[i].emplace_back(result);
            }
        }

        return batch_results;
    }
} // namespace NexLab
#endif