#ifndef __PRE_PROCESS_HPP__
#define __PRE_PROCESS_HPP__
#include "common/logger.hpp"
#include "kernel_function/kernel_function.hpp"
#include "opencv2/opencv.hpp"
namespace NexLab {

    /// @brief 计算padding resize参数
    /// @param src 输入图像
    /// @param params reisze参数
    /// @param tar_w 目标模型输入宽度
    /// @param tar_h 目标模型输入高度
    inline void compute_resize_params(const cv::Mat& src, ResizeParams& params, int tar_w, int tar_h) {
        int image_w = src.cols;
        int image_h = src.rows;

        float scale_w = static_cast<float>(tar_w) / image_w;
        float scale_h = static_cast<float>(tar_h) / image_h;
        float scale = std::min(scale_w, scale_h);

        int resize_w = static_cast<int>(image_w * scale);
        int resize_h = static_cast<int>(image_h * scale);

        int pad_w = (tar_w - resize_w) / 2;
        int pad_h = (tar_h - resize_h) / 2;

        pad_w = std::max(0, pad_w);
        pad_h = std::max(0, pad_h);

        params.image_width = image_w;
        params.image_height = image_h;
        params.ratio = scale;
        params.pad_w = pad_w;
        params.pad_h = pad_h;
        params.resize_w = resize_w;
        params.resize_h = resize_h;
    }

    /// @brief 图像灰度填充
    /// @param img 原始图像
    /// @param resize_img 灰度填充图像
    /// @param model_input_w 模型输入宽度
    /// @param model_input_h 模型输入高度
    /// @param resize_params 填充尺寸
    void inline adaptive_resize(
        const cv::Mat& img, cv::Mat& resize_img, int model_input_w, int model_input_h, ResizeParams& resize_params) {
        float r_w = static_cast<float>(model_input_w) / img.cols;
        float r_h = static_cast<float>(model_input_h) / img.rows;

        resize_params.image_width = img.cols;
        resize_params.image_height = img.rows;
        resize_params.ratio = std::min(r_w, r_h);

        int resize_w = static_cast<int>(img.cols * resize_params.ratio);
        int resize_h = static_cast<int>(img.rows * resize_params.ratio);

        resize_params.pad_w = static_cast<int>((model_input_w - resize_w) / 2);
        resize_params.pad_h = static_cast<int>((model_input_h - resize_h) / 2);
        resize_params.resize_w = resize_w;
        resize_params.resize_h = resize_h;


        cv::Mat resized_img(resize_h, resize_w, CV_8UC3);
        cv::resize(img, resized_img, resized_img.size(), 0, 0, cv::INTER_LINEAR);
        // Image gray scale filling
        resize_img = cv::Mat(model_input_h, model_input_w, CV_8UC3, cv::Scalar(128, 128, 128));
        resized_img.copyTo(
            resize_img(cv::Rect(resize_params.pad_w, resize_params.pad_h, resized_img.cols, resized_img.rows)));
    }

    void inline blob_from_image(
        const cv::Mat& img, std::vector<float>& blob, int channels, int model_input_w, int model_input_h) {
        if (img.empty()) {
            throw std::runtime_error("Input image is empty.");
        }
        if (img.channels() != channels) {
            throw std::runtime_error("Input image channels do not match the sepcified channels");
        }

        blob.resize(channels * model_input_w * model_input_h, 0.0f); // channecl*h*w

        // BGR->RGB and normalize
        for (int row = 0; row < model_input_h; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < model_input_w; ++col) {

                blob[row * model_input_w + col] = static_cast<float>(uc_pixel[2]) / 255.0;

                blob[model_input_h * model_input_w + row * model_input_w + col] =
                    static_cast<float>(uc_pixel[1]) / 255.0;

                blob[2 * model_input_h * model_input_w + row * model_input_w + col] =
                    static_cast<float>(uc_pixel[0]) / 255.0;
                uc_pixel += channels;
            }
        }
    }
} // namespace NexLab
#endif