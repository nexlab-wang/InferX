#ifndef __MODEL_UTILS_HPP__
#define __MODEL_UTILS_HPP__
#include "common/infer_params.hpp"
#include "common/logger.hpp"
#include "opencv2/opencv.hpp"
namespace NexLab {
    void inline decode_class(const float* output_data, int batch_offset, int detection_idx, int detect_num,
        int class_num, int& class_id, float& confidence) {
        for (int i = 0; i < class_num; ++i) {

            float class_score = output_data[batch_offset + (4 + i) * detect_num + detection_idx];

            if (class_score > confidence) {

                confidence = class_score;
                class_id = i;
            }
        }
    }

    void inline decode_bbox(const float* output_data, int batch_offset, int detection_idx, int detect_num,
        const ResizeParams& resize_params, Detection& det) {
        // 1*84*8400类别优先访问
        float cx = output_data[batch_offset + 0 * detect_num + detection_idx];
        float cy = output_data[batch_offset + 1 * detect_num + detection_idx];
        float cw = output_data[batch_offset + 2 * detect_num + detection_idx];
        float ch = output_data[batch_offset + 3 * detect_num + detection_idx];

        // Convert coordinates to original image space
        float x = (cx - resize_params.pad_w) / resize_params.ratio;
        float y = (cy - resize_params.pad_h) / resize_params.ratio;
        float w = cw / resize_params.ratio;
        float h = ch / resize_params.ratio;

        // Calculate integer bounding box coordinates
        int left = std::max(static_cast<int>(x - 0.5f * w), 0);
        int top = std::max(static_cast<int>(y - 0.5f * h), 0);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        det.left = left;
        det.top = top;
        det.width = width;
        det.height = height;
    }

    float inline compute_iou(const std::vector<float>& box1, const std::vector<float>& box2) {
        float x1 = std::max(box1[0], box2[0]);
        float y1 = std::max(box1[1], box2[1]);
        float x2 = std::min(box1[0] + box1[2], box2[0] + box2[2]);
        float y2 = std::min(box1[1] + box1[3], box2[1] + box2[3]);

        float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

        float box1_area = box1[2] * box1[3];
        float box2_area = box2[2] * box2[3];

        return inter_area / (box1_area + box2_area - inter_area);
    }

    bool inline is_in_image(const std::vector<float>& box, float img_width, float img_height) {
        const float x = box[0];
        const float y = box[1];
        const float width = box[2];
        const float height = box[3];

        // Check if the top-left corner is within the image
        if (x < 0 || x > img_width || y < 0 || y > img_height) {
            return false;
        }

        // Check if the bottom-right corner is within the image
        const float right = x + width;
        const float bottom = y + height;
        return right <= img_width && bottom <= img_height;
    }

    float inline mahalanobis_distance(
        const cv::Point2f& mu1, const cv::Mat& sigma1, const cv::Point2f& mu2, const cv::Mat& sigma2) {
        cv::Mat sigma_sum = sigma1 + sigma2;
        cv::Mat sigma_inv;
        cv::invert(sigma_sum, sigma_inv, cv::DECOMP_SVD); // 计算逆矩阵

        cv::Mat delta(2, 1, CV_32F);
        delta.at<float>(0, 0) = mu1.x - mu2.x;
        delta.at<float>(1, 0) = mu1.y - mu2.y;

        cv::Mat dist = delta.t() * sigma_inv * delta;
        return std::sqrt(dist.at<float>(0, 0)); // 计算 Mahalanobis 距离
    }

    float inline compute_prob_iou(const InferRes& res1, const InferRes& res2) {
        // 提取中心点和角度
        cv::Point2f mu1(res1.detection->left, res1.detection->top);
        cv::Point2f mu2(res2.detection->left, res2.detection->top);
        float theta1 = res1.detection->angle;
        float theta2 = res2.detection->angle;

        // 计算协方差矩阵
        cv::Mat R1 = (cv::Mat_<float>(2, 2) << cos(theta1), -sin(theta1), sin(theta1), cos(theta1));
        cv::Mat R2 = (cv::Mat_<float>(2, 2) << cos(theta2), -sin(theta2), sin(theta2), cos(theta2));

        cv::Mat Sigma1 = R1
                       * (cv::Mat_<float>(2, 2) << res1.detection->width * res1.detection->width / 4, 0, 0,
                           res1.detection->height * res1.detection->height / 4)
                       * R1.t();
        cv::Mat Sigma2 = R2
                       * (cv::Mat_<float>(2, 2) << res2.detection->width * res2.detection->width / 4, 0, 0,
                           res2.detection->height * res2.detection->height / 4)
                       * R2.t();

        // 计算 Mahalanobis 距离
        float D = mahalanobis_distance(mu1, Sigma1, mu2, Sigma2);

        // 计算 probIoU
        return exp(-0.5 * D * D);
    }
} // namespace NexLab
#endif