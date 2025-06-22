#ifndef __VIEWER_HPP__
#define __VIEWER_HPP__
#include "infer_params.hpp"
#include "opencv2/opencv.hpp"

#include <vector>

namespace NexLab {
    const std::vector<std::pair<int, int>> POSE_CONNECTIONS = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5},
        {4, 6}};

    inline std::vector<cv::Scalar> generate_class_colors(int num_classes) {
        std::vector<cv::Scalar> colors;

        if (num_classes <= 0)
            return colors;

        const std::vector<cv::Scalar> base_colors = {
            cv::Scalar(255, 0, 0), // 红
            cv::Scalar(0, 255, 0), // 绿
            cv::Scalar(0, 0, 255), // 蓝
            cv::Scalar(255, 255, 0), // 黄
            cv::Scalar(255, 0, 255), // 紫
            cv::Scalar(0, 255, 255), // 青
            cv::Scalar(255, 128, 0), // 橙
            cv::Scalar(128, 0, 255) // 紫罗兰
        };

        if (num_classes <= static_cast<int>(base_colors.size())) {
            colors.assign(base_colors.begin(), base_colors.begin() + num_classes);
            return colors;
        }

        colors = base_colors;
        for (int i = static_cast<int>(base_colors.size()); i < num_classes; ++i) {
            float hue = static_cast<float>(i) * 180.0f / num_classes;
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
            cv::Mat rgb;
            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
            colors.emplace_back(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);
        }

        return colors;
    }

    inline void draw_detection(cv::Mat& frame, const Detection& det, const cv::Scalar& color) {
        if (det.angle >= 0 && det.angle <= CV_PI / 2) {
            cv::Point2f center(det.left + det.width / 2.0f, det.top + det.height / 2.0f);
            cv::Size2f size(det.width, det.height);
            cv::RotatedRect rotated_rect(center, size, det.angle * 180.0f / CV_PI);

            cv::Point2f vertices[4];
            rotated_rect.points(vertices);
            for (int j = 0; j < 4; ++j) {
                cv::line(frame, vertices[j], vertices[(j + 1) % 4], color, 2);
            }
        } else {
            cv::Rect rect(det.left, det.top, det.width, det.height);
            cv::rectangle(frame, rect, color, 2);
        }

        std::string label = cv::format("Class %d: %.2f", det.class_id, det.confidence);
        cv::putText(frame, label, cv::Point(det.left, det.top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    inline void draw_pose(cv::Mat& frame, const Pose& pose) {
        for (int k = 0; k < pose.keypoints.size(); ++k) {
            const auto& kpt = pose.keypoints[k];
            if (kpt.size() >= 2) {
                cv::Scalar kpt_color((k * 50 + 100) % 256, (k * 100 + 50) % 256, (k * 150 + 200) % 256);
                cv::circle(frame, cv::Point(kpt[0], kpt[1]), 5, kpt_color, -1);
            }
        }

        for (int c = 0; c < POSE_CONNECTIONS.size(); ++c) {
            const auto& conn = POSE_CONNECTIONS[c];
            if (conn.first < pose.keypoints.size() && conn.second < pose.keypoints.size()) {
                const auto& pt1 = pose.keypoints[conn.first];
                const auto& pt2 = pose.keypoints[conn.second];
                if (pt1.size() >= 2 && pt2.size() >= 2) {
                    cv::Scalar line_color((c * 70 + 80) % 256, (c * 120 + 30) % 256, (c * 90 + 60) % 256);
                    cv::line(frame, cv::Point(pt1[0], pt1[1]), cv::Point(pt2[0], pt2[1]), line_color, 2);
                }
            }
        }
    }

    inline void draw_segment(cv::Mat& frame, const cv::Mat& mask, const cv::Rect& rect, const cv::Scalar& color) {
        if (!mask.empty()) {
            cv::Mat mask_thresh;
            cv::threshold(mask, mask_thresh, 0.5, 1.0, cv::THRESH_BINARY);

            cv::Mat mask_8u;
            mask_thresh.convertTo(mask_8u, CV_8UC1, 255.0);

            cv::Rect valid_rect = rect & cv::Rect(0, 0, frame.cols, frame.rows);

            if (valid_rect.width <= 0 || valid_rect.height <= 0) {
                std::cerr << "Warning: Invalid rect, skipping mask overlay." << std::endl;
                return;
            }

            if (valid_rect.width != mask_8u.cols || valid_rect.height != mask_8u.rows) {
                std::cerr << "Error: mask size (" << mask_8u.cols << "x" << mask_8u.rows
                          << ") does not match rect size (" << valid_rect.width << "x" << valid_rect.height << ")"
                          << std::endl;
                return;
            }

            cv::Mat colored_mask(valid_rect.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            colored_mask.setTo(color, mask_8u);

            cv::Mat roi = frame(valid_rect);
            cv::addWeighted(roi, 0.7, colored_mask, 0.3, 0, roi);
        }
    }

    inline void viewer(const std::vector<cv::Mat>& batch_frame, const std::vector<std::vector<InferRes>>& batch_result,
        const std::vector<cv::Scalar>& colors) {
        if (colors.empty()) {
            std::cerr << "Error: colors vector is empty!" << std::endl;
            return;
        }

        int delay = 300; // 毫秒
        bool should_exit = false;

        for (size_t i = 0; i < batch_frame.size(); ++i) {
            cv::Mat frame = batch_frame[i].clone();
            const auto& results = batch_result[i];

            for (const auto& res : results) {
                cv::Scalar color = cv::Scalar(0, 255, 0); // 默认绿色
                if (res.detection.has_value()) {
                    int class_id = res.detection.value().class_id;
                    color = colors[class_id % colors.size()];
                } else if (res.obb.has_value()) {
                    int class_id = res.obb.value().class_id;
                    color = colors[class_id % colors.size()];
                }

                if (res.detection.has_value()) {
                    const auto& det = res.detection.value();
                    draw_detection(frame, det, color);

                    if (res.segment_mask.has_value()) {
                        cv::Rect rect(det.left, det.top, det.width, det.height);
                        draw_segment(frame, res.segment_mask.value(), rect, color);
                    }
                }

                if (res.pose.has_value()) {
                    draw_pose(frame, res.pose.value());
                }
            }

            // 显示处理后的帧
            std::string window_name = "Frame " + std::to_string(i);
            cv::imshow(window_name, frame);
            int key = cv::waitKey(delay);

            cv::destroyWindow(window_name);

            if (key == 'q') {
                should_exit = true;
            }
        }

        cv::destroyAllWindows();
    }
} // namespace NexLab
#endif