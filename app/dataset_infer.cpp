#include "common/visualizer.hpp"
#include "inferx.hpp"

#include <iostream>
#include <string>

int dataset_infer() {
    NexLab::InferX inferx;

    std::string config_path = "F:\\Studio\\inferx\\config\\inferx_config.json";

    if (!inferx.load_config(config_path)) {
        LOG_ERROR(NexLab::Logger::GetInstance(), "APP loading infex config is Failed.");
        return -1;
    }

    if (!inferx.init_dataset_stream()) {
        LOG_ERROR(NexLab::Logger::GetInstance(), " APP loading dataset is Failed.");
        return -1;
    }

    const int batch_size = 4;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Scalar> colors;

    auto is_viewer = inferx.is_viewer();
    if (is_viewer)
        colors = NexLab::generate_class_colors(80);

    while (inferx.is_stream_open()) {
        batch_frame.clear();
        cv::Mat frame;

        for (int i = 0; i < batch_size && inferx.get_next_frame(frame); ++i) {
            batch_frame.emplace_back(frame.clone());
        }

        if (batch_frame.empty())
            break;

        std::vector<std::vector<NexLab::InferRes>> batch_result;
        if (inferx.model_infer(batch_frame, batch_result)) {
            if (is_viewer)
                NexLab::viewer(batch_frame, batch_result, colors);
        }
    }

    return 0;
}

int main() {
    dataset_infer();
}