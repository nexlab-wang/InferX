#ifndef __MODEL_BASE__
#define __MODEL_BASE__
#include "common/infer_params.hpp"
#include "common/logger.hpp"
#include "infer_engine/base_engine.hpp"
#include "opencv2/opencv.hpp"
namespace NexLab {
    class ModelBase {
    public:
        ModelBase(std::shared_ptr<BaseEngine> engine, std::shared_ptr<ImageModelParams> params);
        virtual ~ModelBase();

        bool load_model(const std::string& model_file);

        std::vector<std::vector<InferRes>> detection(std::vector<cv::Mat>& images);

        virtual void pre_process_host(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) = 0;

        virtual void post_process_host(
            std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) = 0;

        virtual void pre_process_device(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) = 0;

        virtual void post_process_device(
            std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) = 0;

        std::shared_ptr<BaseEngine> engine_;
        std::shared_ptr<ImageModelParams> params_;

        std::vector<float*> input_buffers_;
        std::vector<float*> output_buffers_;
        std::vector<float*> h_input_buffers_;
        std::vector<float*> h_output_buffers_;

        std::vector<size_t> input_sizes_;
        std::vector<size_t> output_sizes_;
    };
} // namespace NexLab


#endif