#ifndef __INFERX_HPP__
#define __INFERX_HPP__
#include "common/device_info.hpp"
#include "common/infer_params.hpp"
#include "common/logger.hpp"
#include "common/model_param.hpp"
#include "common/param_parser.hpp"
#include "common/utils.hpp"
#include "infer_engine/base_engine.hpp"
#include "infer_engine/openvino_engine.hpp"
#include "infer_engine/tensorrt_engine.hpp"
#include "models/rtdetr.hpp"
#include "models/yolo.hpp"

#include <iostream>

namespace NexLab {
    class InferX {
    private:
        InferXConfig infer_config_;
        std::shared_ptr<BaseEngine> engine_;
        std::shared_ptr<ModelBase> model_;
        std::shared_ptr<ModelParamsBase> model_params_;

        bool is_camera_stream_;
        bool is_video_stream_;
        std::vector<std::string> dataset_files_;
        size_t current_dataset_index_;

    public:
        InferX();

        ~InferX() {
            close_stream();
        };

        bool load_config(std::string config_file_path);
        bool init_dataset_stream();
        bool get_next_frame(cv::Mat& frame);

        bool model_infer(std::vector<cv::Mat>& inputs, std::vector<std::vector<InferRes>>& outputs);

        bool is_stream_open() const;
        void close_stream();

        bool is_viewer() const{
            return infer_config_.enable_visualization;
        }
        
        bool is_dataset_stream() const {
            return !dataset_files_.empty();
        }
    };


} // namespace NexLab
#endif