#ifndef __KERNEL_FUNCTION__
#define __KERNEL_FUNCTION__
#include "common/infer_params.hpp"

#include <cuda_runtime.h>


namespace NexLab {
    void launch_yolo_preprocess_kernel(const uint8_t* d_input, const size_t* d_offsets, const ResizeParams* d_params,
        float* d_output, int batch_size, int model_w, int model_h);

    void launch_rtdetr_preprocess_kernel(const uint8_t* d_input, const size_t* d_offsets, float* d_output,
        int batch_size, int model_w, int model_h, const ResizeParams* d_params);

    void launch_decode_boxes_kernel(const float* d_output_data, const ResizeParams* d_resize_params, int batch_size,
        int class_num, int mask_num, int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices,
        Detection* d_detections, float* d_mask_coeffs, cudaStream_t stream = nullptr);

    void launch_decode_detections_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections);

    void launch_decode_pose_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int key_point_num, int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices,
        Detection* detections, float* key_points);

    void launch_decode_obb_kernel(const float* output, const ResizeParams* resize_params, int batch_size, int class_num,
        int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices, Detection* detections);


    void launch_decode_mask_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        int total_detections, int mask_num, int mask_size, int model_width, int model_height);

    void launch_decode_mask_2stage_kernel(const float* output_mask, const float* d_mask_coeffs, float* d_base_mask,
        float* d_resize_mask, int total_detections, int mask_num, int mask_size, int model_width, int model_height);

    void launch_decode_mask_batch_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        const int* batch_indices, int total_batches, int total_detections, int mask_num, int mask_size, int model_width,
        int model_height);

    void launch_decode_rtdetr_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int detection_num, int class_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections);

}; // namespace NexLab
#endif