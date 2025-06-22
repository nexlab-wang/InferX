#include "common/device_info.hpp"
#include "common/model_param.hpp"
#include "common/param_parser.hpp"
#include "infer_engine/openvino_engine.hpp"
#include "infer_engine/tensorrt_engine.hpp"
#include "models/yolo.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils/test_utils.hpp"

#include <filesystem>
#include <iostream>

void test_detection_openvino_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_detection.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\openvino\\yolov8n_detection\\yolov8n_detection.xml";

    std::string model_path = "F:\\Studio\\inferx\\models\\openvino\\yolov8n.xml";

    auto openvino = std::make_shared<NexLab::OpenVINOEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_CPU, 0);
    openvino->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(openvino), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(80);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            // 批量推理
            auto batch_detects = yolo.detection(batch_images);

            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);

                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
            }
        }
    }
}

void test_detection_tensoorrt_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_detection.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\tensorrt\\yolov8n_detection.trt";

    std::string model_path = "F:\\Studio\\inferx\\models\\tensorrt\\yolov8n.trt";

    auto trt = std::make_shared<NexLab::TensorRTEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    // NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU_CPU, 0);
    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU, 0);
    trt->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(trt), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(80);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            // 批量推理
            auto batch_detects = yolo.detection(batch_images);

            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);

                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
                // int key = cv::waitKey(5);
                // if (key == 'q') {
                //     break;
                // }
            }
        }
    }
}

void test_segment_openvino_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_segment.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco128\\images\\train2017";
    // std::string model_path = "F:\\Datasets\\umt\\model\\openvino\\segment\\yolo_segment.xml";
    std::string model_path = "F:\\Studio\\inferx\\models\\openvino\\yolov8n-seg.xml";


    auto openvino = std::make_shared<NexLab::OpenVINOEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_CPU, 0);
    openvino->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(openvino), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(80);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            // 批量推理
            auto batch_detects = yolo.detection(batch_images);

            // 处理当前批次的检测结果
            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);

                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                // cv::waitKey(0);
                int key = cv::waitKey(5);
                if (key == 'q') {
                    break;
                }
            }
        }
    }
}

void test_segment_tensorrt_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_segment.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco128\\images\\train2017";
    // std::string model_path = "F:\\Datasets\\umt\\model\\tensorrt\\yolov8n_segment.trt";
    std::string model_path = "F:\\Studio\\inferx\\models\\tensorrt\\yolov8n-seg.trt";

    auto trt = std::make_shared<NexLab::TensorRTEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    // NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU_CPU, 0);
    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU, 0);
    trt->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(trt), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(80);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            // 批量推理
            auto batch_detects = yolo.detection(batch_images);

            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {

                    // LOG_INFO(NexLab::Logger::GetInstance(), "detection class id: {}, box: [{} {} {} {}].",
                    //     detect.detection->class_id, detect.detection->left, detect.detection->top,
                    //     detect.detection->width, detect.detection->height);
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);

                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                // cv::waitKey(0);

                int key = cv::waitKey(500);
                if (key == 'q') {
                    break;
                }
            }
        }
    }
}

void test_pose_openvino_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_pose.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8-pose\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\openvino\\yolov8n_pose\\yolov8n_pose.xml";
    std::string model_path = "F:\\Studio\\inferx\\models\\openvino\\yolov8n-pose.xml";

    auto openvino = std::make_shared<NexLab::OpenVINOEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_CPU, 0);
    openvino->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(openvino), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(3);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            auto batch_detects = yolo.detection(batch_images);
            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);
                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
            }
        }
    }
}

void test_pose_tensorrt_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_pose.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8-pose\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\tensorrt\\yolov8n-pose.trt";
    std::string model_path = "F:\\Studio\\inferx\\models\\tensorrt\\yolov8n-pose.trt";

    auto trt = std::make_shared<NexLab::TensorRTEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    // NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU_CPU, 0);
    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU, 0);
    trt->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(trt), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(3);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            auto batch_detects = yolo.detection(batch_images);
            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);
                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
            }
        }
    }
}

void test_obb_openvino_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_obb.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\dota8\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\openvino\\yolov8n_obb\\yolov8n_obb.xml";
    std::string model_path = "F:\\Studio\\inferx\\models\\openvino\\yolov8n-obb.xml";

    auto openvino = std::make_shared<NexLab::OpenVINOEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_CPU, 0);
    openvino->set_device_info(device);
    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(openvino), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(3);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            auto batch_detects = yolo.detection(batch_images);
            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);
                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
            }
        }
    }
}

void test_obb_tensorrt_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\yolo_obb.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\dota8\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\tensorrt\\yolov8n-obb.trt";
    std::string model_path = "F:\\Studio\\inferx\\models\\tensorrt\\yolov8n-obb.trt";

    auto trt = std::make_shared<NexLab::TensorRTEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    // NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU_CPU, 0);
    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU, 0);
    trt->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::YOLO yolo(std::static_pointer_cast<NexLab::BaseEngine>(trt), params);
    yolo.load_model(model_path);

    if (res) {
        std::vector<cv::Scalar> class_colors = NexLab::test_utils::generate_class_colors(3);

        // 按 batch_size 分批处理图像
        for (size_t i = 0; i < images_list.size(); i += params->batch_size) {
            // 获取当前批次的图像
            std::vector<cv::Mat> batch_images;
            for (size_t j = 0; j < params->batch_size && (i + j) < images_list.size(); ++j) {
                auto image = cv::imread(images_list[i + j]);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << images_list[i + j] << std::endl;
                    continue;
                }
                batch_images.push_back(image);
            }

            // 如果当前批次的图像数量不足 batch_size，填充空白图像
            while (batch_images.size() < params->batch_size) {
                batch_images.push_back(cv::Mat::zeros(params->dst_h, params->dst_w, CV_8UC3));
            }

            auto batch_detects = yolo.detection(batch_images);
            for (size_t j = 0; j < batch_detects.size(); ++j) {
                const auto& detects = batch_detects[j];
                auto image = batch_images[j];

                if (image.empty()) {
                    continue;
                }

                for (const auto& detect : detects) {
                    int class_id = detect.detection->class_id;
                    cv::Scalar color =
                        (class_id < class_colors.size()) ? class_colors[class_id] : cv::Scalar(255, 255, 255);
                    NexLab::test_utils::show_infer_res(image, detect, color);
                }

                // 显示检测结果
                cv::imshow("Detection Result", image);
                cv::waitKey(0);
            }
        }
    }
}

int main() {
    // test_detection_openvino_batch();
    // test_detection_tensoorrt_batch();
    // test_segment_openvino_batch();
    // test_segment_tensorrt_batch();
    // test_pose_tensorrt_batch();
    // test_pose_openvino_batch();
    // test_obb_tensorrt_batch();
    test_obb_openvino_batch();
    return 0;
}

// 封装外部接口
// 封装可视化类
// 增加pointnet
// 多batch测试