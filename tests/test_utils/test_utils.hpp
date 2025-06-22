#ifndef __TEST_UTILS_HPP__
#define __TEST_UTILS_HPP__
#include "opencv2/opencv.hpp"

#include <filesystem>
#include <string>

namespace NexLab {
    namespace test_utils {
        /// @brief 读取图像文件列表
        /// @param image_folder --图像文件夹路径
        /// @param image_list --图像列表
        /// @return 若读取成功，返回true，否则返回false.
        bool inline read_image_list(const std::string image_folder, std::vector<std::string>& image_list) {
            if (!std::filesystem::exists(image_folder) || !std::filesystem::is_directory(image_folder)) {
                return false;
            }

            for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().string();
                    if (filename.size() >= 4) {
                        std::string extension = filename.substr(filename.size() - 4);
                        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"
                            || extension == ".tif" || extension == ".tiff") {
                            image_list.push_back(image_folder + "/" + entry.path().filename().string());
                        }
                    }
                }
            }
            return !image_list.empty();
        }


        std::vector<cv::Scalar> inline generate_class_colors(int num_classes) {
            std::vector<cv::Scalar> colors;
            for (int i = 0; i < num_classes; ++i) {
                float hue = static_cast<float>(i) / num_classes * 180.0f;
                cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
                cv::Mat bgr;
                cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                cv::Scalar color(bgr.at<cv::Vec3b>(0, 0)[0], bgr.at<cv::Vec3b>(0, 0)[1], bgr.at<cv::Vec3b>(0, 0)[2]);
                colors.push_back(color);
            }
            return colors;
        }

        void inline show_infer_res(
            cv::Mat& image, const InferRes& res, const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
            Detection res_detection;
            Pose pose;
            OBB obb;
            cv::Mat mask_box;
            bool is_detect = false;

            if (res.detection.has_value()) {
                res_detection = res.detection.value();
                is_detect = true;
            }
            if (res.pose.has_value()) {
                pose = res.pose.value();
            }
            if (res.segment_mask.has_value()) {
                mask_box = res.segment_mask.value();
            }
            if (res.obb.has_value()) {
                obb = res.obb.value();
            }

            cv::Rect rect;

            if (is_detect) {
                float conf_val = res_detection.confidence;
                int class_id = res_detection.class_id;
                int left = res_detection.left;
                int top = res_detection.top;
                int width = res_detection.width;
                int height = res_detection.height;

                if (res_detection.angle >= 0 && res_detection.angle <= CV_PI / 2) {
                    cv::Point2f center(left + width / 2.0f, top + height / 2.0f);
                    cv::Size2f size(width, height);

                    cv::RotatedRect rotated_rect(
                        center, size, res_detection.angle * 180.0f / CV_PI); 
                    cv::Point2f vertices[4];
                    rotated_rect.points(vertices);
                    for (int i = 0; i < 4; ++i) {
                        cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
                    }
                } else {
                    if (left == 0 && top == 0 && width == 0 && height == 0) {
                        std::cerr << "Warning: Invalid rect, rect is 0." << std::endl;
                    }
                    rect.x = left;
                    rect.y = top;
                    rect.width = width;
                    rect.height = height;
                    cv::rectangle(image, rect, color, 2);
                }
                std::string label = "Class ID: " + std::to_string(class_id) + " Conf: " + std::to_string(conf_val);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }

            if (!pose.keypoints.empty()) {
                const std::vector<std::pair<int, int>> connections = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                    {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
                    {3, 5}, {4, 6}};

                for (int i = 0; i < pose.keypoints.size(); ++i) {
                    auto keypoint = pose.keypoints[i];
                    int blue = (i * 50 + 100) % 256;
                    int green = (i * 100 + 50) % 256;
                    int red = (i * 150 + 200) % 256;
                    cv::Scalar color_pose(blue, green, red);
                    cv::circle(image, cv::Point(keypoint[0], keypoint[1]), 3, color_pose, -1);
                }

                for (const auto& conn : connections) {
                    int start_idx = conn.first;
                    int end_idx = conn.second;

                    if (start_idx < pose.keypoints.size() && end_idx < pose.keypoints.size()) {
                        auto start_point = pose.keypoints[start_idx];
                        auto end_point = pose.keypoints[end_idx];

                        if (start_point[0] > 0 && start_point[1] > 0 && end_point[0] > 0 && end_point[1] > 0) {
                            cv::line(image, cv::Point(start_point[0], start_point[1]),
                                cv::Point(end_point[0], end_point[1]), cv::Scalar(255, 255, 0), 2);
                        }
                    }
                }
            }

            if (!mask_box.empty()) {

                cv::Mat mask_thresh;
                cv::threshold(mask_box, mask_thresh, 0.5, 1.0, cv::THRESH_BINARY);

                cv::Mat mask_8u;
                mask_thresh.convertTo(mask_8u, CV_8UC1, 255.0);


                rect = rect & cv::Rect(0, 0, image.cols, image.rows);


                if (rect.width <= 0 || rect.height <= 0) {

                    std::cerr << "Warning: Invalid rect, skipping mask overlay." << std::endl;
                } else {

                    if (rect.width == mask_8u.cols && rect.height == mask_8u.rows) {

                        cv::Mat colored_mask_roi(rect.size(), CV_8UC3, cv::Scalar(0, 0, 0));
                        colored_mask_roi.setTo(color, mask_8u);

                        cv::Mat roi = image(rect);
                        cv::addWeighted(roi, 0.7, colored_mask_roi, 0.3, 0, roi);
                    } else {
                        std::cerr << "Error: mask_8u size does not match rect size." << std::endl;
                    }
                }
            }
        }

    } // namespace test_utils
} // namespace NexLab

#endif