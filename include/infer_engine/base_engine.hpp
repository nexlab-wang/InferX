#ifndef __BASE_ENGINE_HPP__
#define __BASE_ENGINE_HPP__

#include "common/device_info.hpp"
#include "common/dllexport.hpp"
#include "common/logger.hpp"
#include "common/model_param.hpp"
#include "engine_info.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace NexLab {
    class MODEL_INFER_API BaseEngine {
    private:
        std::vector<float*> input_src_datas_;
        std::vector<float*> output_src_datas_;

    public:
        BaseEngine(/* args */) {
            init_logger();
        };

        virtual ~BaseEngine() {};

        virtual bool load_dev() = 0;

        virtual bool load_model(const std::string& model) = 0;

        virtual bool infer_model() = 0;

        virtual void checking() = 0;

        virtual const std::unordered_map<std::string, ioDims>& get_io_dims() const = 0;

        void initialize(const ModelParamsBase& params);

        void set_device_info(const DeviceInfo& dev_info);

        const DeviceInfo& get_device_info() const;

        void set_input_datas(std::vector<float*> inputs);

        const std::vector<float*> get_input_datas() const;

        void set_output_datas(std::vector<float*> outputs);

        const std::vector<float*> get_output_datas() const;

        DeviceInfo dev_info_;
        std::shared_ptr<const ModelParamsBase> b_params_;
        std::shared_ptr<const ImageModelParams> b_image_params_;
    };
} // namespace NexLab


#endif