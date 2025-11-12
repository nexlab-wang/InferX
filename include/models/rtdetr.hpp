#ifndef __RTDETR_HPP__
#define __RTDETR_HPP__
#include "common/dllexport.hpp"
#include "models/model_base.hpp"
namespace NexLab {
    class MODEL_INFER_API RTDETR : public ModelBase {
    private:
        /* data */
    public:
        RTDETR(std::shared_ptr<BaseEngine> engine, std::shared_ptr<ImageModelParams> params)
            : ModelBase(engine, params) {};
        ~RTDETR() {};

        void pre_process_host(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) override;

        void post_process_host(
            std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) override;

        void pre_process_device(std::vector<cv::Mat>& images, std::vector<ResizeParams>& img_resize_params) override;

        void post_process_device(
            std::vector<std::vector<InferRes>>& outputs, std::vector<ResizeParams>& img_resize_params) override;
    };

} // namespace NexLab

#endif