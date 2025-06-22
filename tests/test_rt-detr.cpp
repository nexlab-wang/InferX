#include "common/device_info.hpp"
#include "common/inferx_config.hpp"
#include "common/model_param.hpp"
#include "common/param_parser.hpp"
#include "infer_engine/openvino_engine.hpp"
#include "infer_engine/tensorrt_engine.hpp"
#include "models/rtdetr.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils/test_utils.hpp"

#include <filesystem>
#include <iostream>


void test_trdetr_tensorrt_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\rt-detr.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8\\images\\train";
    // std::string model_path = "F:\\Datasets\\umt\\model\\tensorrt\\rtdetr-l.trt";
    std::string model_path = "F:\\Studio\\inferx\\models\\tensorrt\\rtdetr-l.trt";

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

    NexLab::RTDETR rt_detr(std::static_pointer_cast<NexLab::BaseEngine>(trt), params);
    rt_detr.load_model(model_path);

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

            auto batch_detects = rt_detr.detection(batch_images);
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
                int key = cv::waitKey(500);
                if (key == 'q') {
                    break;
                }
            }
        }
    }
}

void test_trdetr_openvino_batch() {
    std::string param_path = "F:\\Studio\\inferx\\config\\rt-detr.json";
    std::string images_folder = "F:\\Studio\\Code\\DeepLearning\\ultralytics\\datasets\\coco8\\images\\train";
    std::string model_path = "F:\\Studio\\inferx\\models\\openvino\\rtdetr-l.xml";

    auto openvino = std::make_shared<NexLab::OpenVINOEngine>();
    auto params = std::make_shared<NexLab::ImageModelParams>();

    NexLab::DeviceInfo device(NexLab::DeviceType::DEV_CPU, 0);
    // NexLab::DeviceInfo device(NexLab::DeviceType::DEV_GPU, 0);
    openvino->set_device_info(device);

    NexLab::ParamParser& parser = NexLab::ParamParser::GetInstance();

    if (parser.read_params(param_path, *params)) {
        std::cout << "Parameters read successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read parameters from file." << std::endl;
    }

    std::vector<std::string> images_list;
    auto res = NexLab::test_utils::read_image_list(images_folder, images_list);

    NexLab::RTDETR rt_detr(std::static_pointer_cast<NexLab::BaseEngine>(openvino), params);
    rt_detr.load_model(model_path);

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

            auto batch_detects = rt_detr.detection(batch_images);
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
                int key = cv::waitKey(500);
                if (key == 'q') {
                    break;
                }
            }
        }
    }
}

int main() {
    test_trdetr_tensorrt_batch();
    // test_trdetr_openvino_batch();
    return 0;
}
