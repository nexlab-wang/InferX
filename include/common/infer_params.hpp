#ifndef __INFER_PARAMS__
#define __INFER_PARAMS__
#include "opencv2/opencv.hpp"

#include <cmath>
#include <numeric>
#include <optional>
#include <variant>
#include <vector>

namespace NexLab {
    struct ResizeParams {
        float ratio = 0.f; // 缩放比例
        int pad_w = 0; // 宽度方向的填充
        int pad_h = 0; // 高度方向的填充

        int image_width;
        int image_height;
        int resize_w;
        int resize_h;
    };

    struct Detection {
        float left, top, width, height;
        float angle = -1;
        float confidence = -1;
        int class_id = -1;
    };

    struct Pose {
        std::vector<std::vector<float>> keypoints; // 关键点坐标
    };

    struct OBB {
        float cx, cy, w, h, angle; // 中心点坐标、宽度、高度、旋转角度
        float confidence; // 置信度
        int class_id; // 类别 ID
    };

    struct InferRes {
        std::optional<Detection> detection;
        std::optional<Pose> pose;
        std::optional<OBB> obb; //可以删除掉
        std::optional<cv::Mat> segment_mask;
        std::optional<std::vector<float>> mask_coefficient;

        InferRes() = default;

        InferRes(const Detection& det) : detection(det) {};

        InferRes(const Detection& det, const Pose& pos) : detection(det), pose(pos) {};

        InferRes(const Detection& det, const std::vector<float>& mask_coef, const cv::Mat& mask)
            : detection(det), mask_coefficient(mask_coef), segment_mask(mask) {};

        InferRes(const OBB& ob) : obb(ob) {};
    };
}; // namespace NexLab

#endif