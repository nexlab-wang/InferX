#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel_function/kernel_function.hpp"

#include <float.h>
#include <math.h>

namespace NexLab {


    __device__ float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    __device__ uchar3 bilinear_interpolate(const uint8_t* src, int src_w, int src_h, float x, float y) {
        int x1 = __float2int_rd(x);
        int y1 = __float2int_rd(y);
        int x2 = min(x1 + 1, src_w - 1);
        int y2 = min(y1 + 1, src_h - 1);

        float dx = x - x1;
        float dy = y - y1;

        // 读取四个点的BGR值
        int idx1 = (y1 * src_w + x1) * 3;
        int idx2 = (y1 * src_w + x2) * 3;
        int idx3 = (y2 * src_w + x1) * 3;
        int idx4 = (y2 * src_w + x2) * 3;

        float b = (1 - dx) * (1 - dy) * src[idx1] + dx * (1 - dy) * src[idx2] + (1 - dx) * dy * src[idx3]
                + dx * dy * src[idx4];
        float g = (1 - dx) * (1 - dy) * src[idx1 + 1] + dx * (1 - dy) * src[idx2 + 1] + (1 - dx) * dy * src[idx3 + 1]
                + dx * dy * src[idx4 + 1];
        float r = (1 - dx) * (1 - dy) * src[idx1 + 2] + dx * (1 - dy) * src[idx2 + 2] + (1 - dx) * dy * src[idx3 + 2]
                + dx * dy * src[idx4 + 2];

        return make_uchar3(__float2int_rn(b), __float2int_rn(g), __float2int_rn(r));
    }

    // class first
    __device__ void decode_class_cf_device(const float* data, int batch_offset, int detection_idx, int detection_num,
        int class_num, int& class_id, float& max_score) {
        for (int k = 0; k < class_num; ++k) {
            float score = data[batch_offset + (4 + k) * detection_num + detection_idx];

            if (score > max_score) {
                max_score = score;
                class_id = k;
            }
        }

        return;
    }

    // bboxes first
    __device__ void decode_class_bf_device(
        const float* data, int batch_offset, int class_num, int& class_id, float& max_score) {
        for (int k = 0; k < class_num; ++k) {
            float score = data[4 + k];

            if (score > max_score) {
                max_score = score;
                class_id = k;
            }
        }
        // printf("batch_offset=%d, max_score=%.2f\n", batch_offset, max_score);
        return;
    }

    // class first
    __device__ void decode_bbox_cf_device(const float* data, int batch_offset, int detection_num, int detection_idx,
        const ResizeParams& resize_params, Detection& detection) {
        float cx = data[batch_offset + 0 * detection_num + detection_idx];
        float cy = data[batch_offset + 1 * detection_num + detection_idx];
        float cw = data[batch_offset + 2 * detection_num + detection_idx];
        float ch = data[batch_offset + 3 * detection_num + detection_idx];

        float ratio = resize_params.ratio;
        int pad_w = resize_params.pad_w;
        int pad_h = resize_params.pad_h;

        float x = (cx - pad_w) / ratio;
        float y = (cy - pad_h) / ratio;
        float w = cw / ratio;
        float h = ch / ratio;

        int left = max(static_cast<int>(x - 0.5f * w), 0);
        int top = max(static_cast<int>(y - 0.5f * h), 0);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        if (width <= 0 || height <= 0)
            return;

        detection.left = left;
        detection.top = top;
        detection.width = width;
        detection.height = height;

        return;
    }

    // bboxes first
    __device__ void decode_bbox_bf_device(
        const float* data, int batch_offset, const ResizeParams& resize_params, Detection& detection) {
        int image_width = resize_params.image_width;
        int image_height = resize_params.image_height;

        float cx = data[0];
        float cy = data[1];
        float cw = data[2];
        float ch = data[3];

        int xmin = max(static_cast<int>((cx - 0.5 * cw) * image_width), 0);
        int ymin = max(static_cast<int>((cy - 0.5 * ch) * image_height), 0);
        int xmax = min(static_cast<int>((cx + 0.5 * cw) * image_width), image_width - 1);
        int ymax = min(static_cast<int>((cy + 0.5 * ch) * image_height), image_height - 1);

        int width = xmax - xmin;
        int height = ymax - ymin;

        if (width <= 0 || height <= 0)
            return;

        detection.left = xmin;
        detection.top = ymin;
        detection.width = width;
        detection.height = height;

        return;
    }

    __global__ void padding_bilinear_bgr2rbg_norm_kernel(const uint8_t* input_images, const size_t* image_offsets,
        const ResizeParams* resize_params, float* output_blob, int model_input_w, int model_input_h, int batch_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int batch = blockIdx.z;

        if (x >= model_input_w || y >= model_input_h || batch >= batch_size)
            return;

        ResizeParams params = resize_params[batch];
        int image_w = params.image_width;
        int image_h = params.image_height;
        int pad_w = params.pad_w;
        int pad_h = params.pad_h;
        int resize_w = params.resize_w;
        int resize_h = params.resize_h;
        float ratio = params.ratio;

        // 计算当前像素在resize图像中的位置
        int dst_x = x;
        int dst_y = y;

        uchar3 pixel_value;
        if (dst_x >= pad_w && dst_x < pad_w + resize_w && dst_y >= pad_h && dst_y < pad_h + resize_h) {
            // 计算在原图中的位置
            float src_x = (dst_x - pad_w) / ratio;
            float src_y = (dst_y - pad_h) / ratio;

            const uint8_t* image_data = input_images + image_offsets[batch];
            pixel_value = bilinear_interpolate(image_data, image_w, image_h, src_x, src_y);
        } else {
            // 区域填充
            pixel_value = make_uchar3(128, 128, 128);
        }

        // BGR转RGB并归一化
        float r = static_cast<float>(pixel_value.z) / 255.0f;
        float g = static_cast<float>(pixel_value.y) / 255.0f;
        float b = static_cast<float>(pixel_value.x) / 255.0f;

        int channel_size = model_input_h * model_input_w;
        int batch_offset = batch * 3 * channel_size;

        output_blob[batch_offset + y * model_input_w + x] = r;
        output_blob[batch_offset + channel_size + y * model_input_w + x] = g;
        output_blob[batch_offset + 2 * channel_size + y * model_input_w + x] = b;
    }

    __global__ void resize_bilinear_bgr2rgb_norm_kernel(const uint8_t* input_images, const size_t* image_offsets,
        float* output_blob, int model_input_w, int model_input_h, int batch_size, const ResizeParams* resize_params) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int batch = blockIdx.z;

        if (x >= model_input_w || y >= model_input_h || batch >= batch_size)
            return;

        ResizeParams params = resize_params[batch];
        int image_w = params.image_width;
        int image_h = params.image_height;

        float src_x = (x + 0.5f) * (static_cast<float>(image_w) / model_input_w) - 0.5f;
        float src_y = (y + 0.5f) * (static_cast<float>(image_h) / model_input_h) - 0.5f;

        src_x = max(0.0f, min(src_x, image_w - 1.0f));
        src_y = max(0.0f, min(src_y, image_h - 1.0f));

        const uint8_t* image_data = input_images + image_offsets[batch];
        uchar3 pixel_value = bilinear_interpolate(image_data, image_w, image_h, src_x, src_y);
        
        //norm
        float r = static_cast<float>(pixel_value.z) / 255.0f;
        float g = static_cast<float>(pixel_value.y) / 255.0f;
        float b = static_cast<float>(pixel_value.x) / 255.0f;

        int channel_size = model_input_h * model_input_w;
        int batch_offset = batch * 3 * channel_size;
        int pixel_offset = y * model_input_w + x;
        //bgr->rgb
        output_blob[batch_offset + pixel_offset] = r;
        output_blob[batch_offset + channel_size + pixel_offset] = g;
        output_blob[batch_offset + 2 * channel_size + pixel_offset] = b;
    }

    __global__ void decode_pose_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int key_point_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices,
        Detection* detections, float* key_points) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * detect_num)
            return;

        int batch_idx = idx / detect_num;
        int detect_idx = idx % detect_num;
        int feat_num = 4 + class_num + key_point_num*3;

        if (batch_idx >= batch_size)
            return;

        Detection detect;

        float max_score = -1.0f;
        int class_id = -1;

        int batch_offset = batch_idx * feat_num * detect_num;

        decode_class_cf_device(output, batch_offset, detect_idx, detect_num, class_num, class_id, max_score);

        if (max_score <= conf_th)
            return;

        decode_bbox_cf_device(output, batch_offset, detect_num, detect_idx, resize_params[batch_idx], detect);

        for (int k = 0; k < key_point_num; k++) {
            float kp_x = output[batch_offset + (4 + class_num + k * 3) * detect_num + detect_idx];
            float kp_y = output[batch_offset + (4 + class_num + k * 3 + 1) * detect_num + detect_idx];
            float kp_conf = output[batch_offset + (4 + class_num + k * 3 + 2) * detect_num + detect_idx];

            key_points[idx * key_point_num * 3 + k * 3] =
                (kp_x - resize_params[batch_idx].pad_w) / resize_params[batch_idx].ratio;
            key_points[idx * key_point_num * 3 + k * 3 + 1] =
                (kp_y - resize_params[batch_idx].pad_h) / resize_params[batch_idx].ratio;
            key_points[idx * key_point_num * 3 + k * 3 + 2] = kp_conf;
        }

        detect.class_id = class_id;
        detect.confidence = max_score;

        detections[idx] = detect;

        int count = atomicAdd(&valid_counts[batch_idx], 1);
        valid_indices[batch_idx * detect_num + count] = idx;
    }

    __global__ void decode_obb_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * detect_num)
            return;

        int batch_idx = idx / detect_num;
        int detect_idx = idx % detect_num;
        int feat_num = 5 + class_num;

        if (batch_idx >= batch_size)
            return;

        Detection detect;

        float max_score = -1.0f;
        int class_id = -1;

        int batch_offset = batch_idx * feat_num * detect_num;

        decode_class_cf_device(output, batch_offset, detect_idx, detect_num, class_num, class_id, max_score);

        if (max_score <= conf_th)
            return;

        decode_bbox_cf_device(output, batch_offset, detect_num, detect_idx, resize_params[batch_idx], detect);

        float angle = output[batch_offset + (feat_num - 1) * detect_num + detect_idx];

        detect.class_id = class_id;
        detect.confidence = max_score;
        detect.angle = angle;

        detections[idx] = detect;

        int count = atomicAdd(&valid_counts[batch_idx], 1);
        valid_indices[batch_idx * detect_num + count] = idx;
    }

    __global__ void decode_detection_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * detect_num)
            return;

        int batch_idx = idx / detect_num;
        int detect_idx = idx % detect_num;
        int feat_num = 4 + class_num;

        if (batch_idx >= batch_size)
            return;

        Detection detect;

        float max_score = -1.0f;
        int class_id = -1;

        int batch_offset = batch_idx * feat_num * detect_num;

        decode_class_cf_device(output, batch_offset, detect_idx, detect_num, class_num, class_id, max_score);

        if (max_score <= conf_th)
            return;

        decode_bbox_cf_device(output, batch_offset, detect_num, detect_idx, resize_params[batch_idx], detect);

        detect.class_id = class_id;
        detect.confidence = max_score;
        detections[idx] = detect;

        int count = atomicAdd(&valid_counts[batch_idx], 1);
        valid_indices[batch_idx * detect_num + count] = idx;
    }

    __global__ void decode_boxes_kernel(const float* output_data, const ResizeParams* resize_params, int batch_size,
        int class_num, int mask_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices,
        Detection* detections, float* mask_coeffs) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * detect_num)
            return;

        int batch_idx = idx / detect_num; // batch size
        int detect_idx = idx % detect_num; // detection index
        int feat_num = 4 + class_num + mask_num;

        if (batch_idx >= batch_size)
            return;

        Detection detect;

        float max_score = -1.0f;
        int class_id = -1;

        int batch_offset = batch_idx * feat_num * detect_num;

        decode_class_cf_device(output_data, batch_offset, detect_idx, detect_num, class_num, class_id, max_score);

        if (max_score <= conf_th)
            return;

        decode_bbox_cf_device(output_data, batch_offset, detect_num, detect_idx, resize_params[batch_idx], detect);

        for (int k = 0; k < mask_num; ++k) {
            mask_coeffs[(batch_idx * detect_num + detect_idx) * mask_num + k] =
                output_data[batch_offset + (4 + class_num + k) * detect_num + detect_idx];
        }

        detect.class_id = class_id;
        detect.confidence = max_score;

        detections[idx] = detect;

        int count = atomicAdd(&valid_counts[batch_idx], 1);
        valid_indices[batch_idx * detect_num + count] = idx;
    }

    __global__ void decode_rtdetr_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int detection_num, int class_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= batch_size * detection_num)
            return;

        int batch_idx = idx / detection_num;
        int detection_idx = idx % detection_num;
        int feat_num = 4 + class_num;

        if (batch_idx >= batch_size || detection_idx >= detection_num)
            return;

        Detection detection;

        float max_score = -1.0f;
        int class_id = -1;

        int batch_offset = batch_idx * feat_num * detection_num + detection_idx * feat_num;
        const float* detect_data = &output[batch_offset];

        decode_class_bf_device(detect_data, batch_offset, class_num, class_id, max_score);

        if (max_score <= conf_th)
            return;

        decode_bbox_bf_device(detect_data, batch_offset, resize_params[batch_idx], detection);

        detection.class_id = class_id;
        detection.confidence = max_score;

        detections[idx] = detection;

        int count = atomicAdd(&valid_counts[batch_idx], 1);
        valid_indices[batch_idx * detection_num + count] = idx;
    }

    // mask解码函数
    __global__ void decode_mask_batch_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        const int* batch_indices, int total_batches, int total_detections, int mask_num, int mask_size, int model_width,
        int model_height) {

        // 3D grid: x->width, y->height, z->detections index
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int det_idx = blockIdx.z;

        if (x >= model_width || y >= model_height || det_idx >= total_detections) {
            return;
        }

        if (output_mask == nullptr || mask_coeffs == nullptr || resize_mask == nullptr || batch_indices == nullptr) {
            return;
        }

        const int batch_idx = batch_indices[det_idx];
        if (batch_idx < 0 || batch_idx >= total_batches) {
            return;
        }

        const float* curr_coeff = mask_coeffs + det_idx * mask_num;

        const float scale_x = static_cast<float>(mask_size) / model_width;
        const float scale_y = static_cast<float>(mask_size) / model_height;
        const float src_x = (x + 0.5f) * scale_x - 0.5f;
        const float src_y = (y + 0.5f) * scale_y - 0.5f;

        const int x_low = floorf(src_x);
        const int y_low = floorf(src_y);
        const int x_high = min(x_low + 1, mask_size - 1);
        const int y_high = min(y_low + 1, mask_size - 1);

        const float lx = src_x - x_low;
        const float ly = src_y - y_low;
        const float hx = 1.0f - lx;
        const float hy = 1.0f - ly;

        float interpolated[4] = {0};
        for (int k = 0; k < mask_num; ++k) {
            const float* proto = output_mask + (batch_idx * mask_num + k) * mask_size * mask_size;
            const float coeff = curr_coeff[k];

            interpolated[0] += coeff * proto[y_low * mask_size + x_low];
            interpolated[1] += coeff * proto[y_low * mask_size + x_high];
            interpolated[2] += coeff * proto[y_high * mask_size + x_low];
            interpolated[3] += coeff * proto[y_high * mask_size + x_high];
        }

        float activated[4];
        for (int i = 0; i < 4; ++i) {
            activated[i] = 1.0f / (1.0f + expf(-interpolated[i]));
        }

        const float final_value =
            activated[0] * hx * hy + activated[1] * lx * hy + activated[2] * hx * ly + activated[3] * lx * ly;

        const int out_idx = det_idx * model_width * model_height + y * model_width + x;
        resize_mask[out_idx] = final_value;
    }

    __global__ void decode_mask_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        int total_detections, int mask_num, int mask_size, int model_width, int model_height) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int det_idx = blockIdx.z;

        if (x >= model_width || y >= model_height || det_idx >= total_detections) {
            return;
        }

        const float* curr_coeff = mask_coeffs + det_idx * mask_num;

        const float scale_x = static_cast<float>(mask_size) / model_width;
        const float scale_y = static_cast<float>(mask_size) / model_height;
        const float src_x = (x + 0.5f) * scale_x - 0.5f;
        const float src_y = (y + 0.5f) * scale_y - 0.5f;

        const int x_low = floorf(src_x);
        const int y_low = floorf(src_y);
        const int x_high = min(x_low + 1, mask_size - 1);
        const int y_high = min(y_low + 1, mask_size - 1);

        const float lx = src_x - x_low;
        const float ly = src_y - y_low;
        const float hx = 1.0f - lx;
        const float hy = 1.0f - ly;

        float interpolated[4] = {0}; // [v1, v2, v3, v4]
        for (int k = 0; k < mask_num; ++k) {
            const float* proto = output_mask + k * mask_size * mask_size;
            const float coeff = curr_coeff[k];

            interpolated[0] += coeff * proto[y_low * mask_size + x_low]; // v1
            interpolated[1] += coeff * proto[y_low * mask_size + x_high]; // v2
            interpolated[2] += coeff * proto[y_high * mask_size + x_low]; // v3
            interpolated[3] += coeff * proto[y_high * mask_size + x_high]; // v4
        }

        float activated[4];
        activated[0] = 1.0f / (1.0f + expf(-interpolated[0]));
        activated[1] = 1.0f / (1.0f + expf(-interpolated[1]));
        activated[2] = 1.0f / (1.0f + expf(-interpolated[2]));
        activated[3] = 1.0f / (1.0f + expf(-interpolated[3]));

        const float final_value =
            activated[0] * hx * hy + activated[1] * lx * hy + activated[2] * hx * ly + activated[3] * lx * ly;

        const int out_idx = det_idx * model_width * model_height + y * model_width + x;
        resize_mask[out_idx] = final_value;
    }


    __global__ void compute_mask_kernel(const float* output_mask, const float* mask_coeffs, float* d_base_mask,
        int mask_num, int mask_size, int total_detections) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int det_idx = blockIdx.z;

        if (x >= mask_size || y >= mask_size || det_idx >= total_detections) {
            return;
        }

        const float* curr_coeff = mask_coeffs + det_idx * mask_num;
        float sum = 0.0f;


        for (int k = 0; k < mask_num; ++k) {
            sum += curr_coeff[k] * output_mask[k * mask_size * mask_size + y * mask_size + x];
        }

        const float activated = 1.0f / (1.0f + expf(-sum));

        const int out_idx = det_idx * mask_size * mask_size + y * mask_size + x;
        d_base_mask[out_idx] = activated;
    }

    __global__ void resize_mask_kernel(const float* d_base_mask, float* resize_mask, int mask_size, int model_width,
        int model_height, int total_detections) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int det_idx = blockIdx.z;

        if (x >= model_width || y >= model_height || det_idx >= total_detections) {
            return;
        }

        const float scale_x = static_cast<float>(mask_size) / model_width;
        const float scale_y = static_cast<float>(mask_size) / model_height;
        const float src_x = (x + 0.5f) * scale_x - 0.5f;
        const float src_y = (y + 0.5f) * scale_y - 0.5f;

        const int x_low = floorf(src_x);
        const int y_low = floorf(src_y);
        const int x_high = min(x_low + 1, mask_size - 1);
        const int y_high = min(y_low + 1, mask_size - 1);

        const float lx = src_x - x_low;
        const float ly = src_y - y_low;
        const float hx = 1.0f - lx;
        const float hy = 1.0f - ly;

        const float* curr_mask = d_base_mask + det_idx * mask_size * mask_size;
        const float v1 = curr_mask[y_low * mask_size + x_low]; // (x_low, y_low)
        const float v2 = curr_mask[y_low * mask_size + x_high]; // (x_high, y_low)
        const float v3 = curr_mask[y_high * mask_size + x_low]; // (x_low, y_high)
        const float v4 = curr_mask[y_high * mask_size + x_high]; // (x_high, y_high)

        const float final_value = v1 * hx * hy + v2 * lx * hy + v3 * hx * ly + v4 * lx * ly;

        const int out_idx = det_idx * model_width * model_height + y * model_width + x;
        resize_mask[out_idx] = final_value;
    }


    void launch_yolo_preprocess_kernel(const uint8_t* d_input, const size_t* d_offsets, const ResizeParams* d_params,
        float* d_output, int batch_size, int model_w, int model_h) {
        dim3 block(16, 16);
        dim3 grid((model_w + block.x - 1) / block.x, (model_h + block.y - 1) / block.y, batch_size);

        padding_bilinear_bgr2rbg_norm_kernel<<<grid, block>>>(
            d_input, d_offsets, d_params, d_output, model_w, model_h, batch_size);

        cudaDeviceSynchronize();
    }

    void launch_rtdetr_preprocess_kernel(const uint8_t* d_input, const size_t* d_offsets, float* d_output,
        int batch_size, int model_w, int model_h, const ResizeParams* d_params) {
        dim3 block(16, 16);
        dim3 grid((model_w + block.x - 1) / block.x, (model_h + block.y - 1) / block.y, batch_size);

        resize_bilinear_bgr2rgb_norm_kernel<<<grid, block>>>(
            d_input, d_offsets, d_output, model_w, model_h, batch_size, d_params);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "RTDETR CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void launch_decode_boxes_kernel(const float* d_output_data, const ResizeParams* d_resize_params, int batch_size,
        int class_num, int mask_num, int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices,
        Detection* d_detections, float* d_mask_coeffs, cudaStream_t stream) {
        dim3 block(256);
        dim3 grid((batch_size * detect_num + block.x - 1) / block.x);

        decode_boxes_kernel<<<grid, block, 0, stream>>>(d_output_data, d_resize_params, batch_size, class_num, mask_num,
            detect_num, conf_th, d_valid_counts, d_valid_indices, d_detections, d_mask_coeffs);

        cudaError_t err = cudaDeviceSynchronize();
    }


    void launch_decode_mask_batch_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        const int* batch_indices, int total_batches, int total_detections, int mask_num, int mask_size, int model_width,
        int model_height) {
        dim3 block(16, 16);
        dim3 grid((model_width + block.x - 1) / block.x, (model_height + block.y - 1) / block.y, total_detections);

        decode_mask_batch_kernel<<<grid, block>>>(output_mask, mask_coeffs, resize_mask, batch_indices, total_batches,
            total_detections, mask_num, mask_size, model_width, model_height);

        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "Kernel launch failed: " + std::string(cudaGetErrorString(kernelErr)) << std::endl;
            // 清理资源
            return;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }


    void launch_decode_mask_kernel(const float* output_mask, const float* mask_coeffs, float* resize_mask,
        int total_detections, int mask_num, int mask_size, int model_width, int model_height) {

        dim3 block(16, 16);
        dim3 grid((model_width + block.x - 1) / block.x, (model_height + block.y - 1) / block.y, total_detections);

        decode_mask_kernel<<<grid, block>>>(
            output_mask, mask_coeffs, resize_mask, total_detections, mask_num, mask_size, model_width, model_height);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }


    void launch_decode_mask_2stage_kernel(const float* output_mask, const float* d_mask_coeffs, float* d_base_mask,
        float* d_resize_mask, int total_detections, int mask_num, int mask_size, int model_width, int model_height) {

        dim3 block1(16, 16);
        dim3 grid1((mask_size + block1.x - 1) / block1.x, (mask_size + block1.y - 1) / block1.y, total_detections);
        compute_mask_kernel<<<grid1, block1>>>(
            output_mask, d_mask_coeffs, d_base_mask, mask_num, mask_size, total_detections);

        dim3 block2(16, 16);
        dim3 grid2((model_width + block2.x - 1) / block2.x, (model_height + block2.y - 1) / block2.y, total_detections);
        resize_mask_kernel<<<grid2, block2>>>(
            d_base_mask, d_resize_mask, mask_size, model_width, model_height, total_detections);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void launch_decode_pose_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int key_point_num, int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices,
        Detection* detections, float* key_points) {
        dim3 block(256);
        dim3 grid((batch_size * detect_num + block.x - 1) / block.x);

        decode_pose_kernel<<<block, grid>>>(output, resize_params, batch_size, class_num, key_point_num, detect_num,
            conf_th, d_valid_counts, d_valid_indices, detections, key_points);


        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }


    void launch_decode_obb_kernel(const float* output, const ResizeParams* resize_params, int batch_size, int class_num,
        int detect_num, float conf_th, int* d_valid_counts, int* d_valid_indices, Detection* detections) {

        dim3 block(256);
        dim3 grid((batch_size * detect_num + block.x - 1) / block.x);

        decode_obb_kernel<<<block, grid>>>(output, resize_params, batch_size, class_num, detect_num, conf_th,
            d_valid_counts, d_valid_indices, detections);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void launch_decode_detections_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int class_num, int detect_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections) {
        dim3 block(256);
        dim3 grid((batch_size * detect_num + block.x - 1) / block.x);

        decode_detection_kernel<<<block, grid>>>(
            output, resize_params, batch_size, class_num, detect_num, conf_th, valid_counts, valid_indices, detections);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void launch_decode_rtdetr_kernel(const float* output, const ResizeParams* resize_params, int batch_size,
        int detection_num, int class_num, float conf_th, int* valid_counts, int* valid_indices, Detection* detections) {
        dim3 block(256);
        dim3 grid((batch_size * detection_num + block.x - 1) / block.x);

        decode_rtdetr_kernel<<<block, grid>>>(output, resize_params, batch_size, detection_num, class_num, conf_th,
            valid_counts, valid_indices, detections);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
        }
    }
}; // namespace NexLab