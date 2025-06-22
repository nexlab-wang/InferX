#ifndef __OPENVINO_HPP__
#define __OPENVINO_HPP__
#include "base_engine.hpp"

#include <openvino/openvino.hpp>
namespace NexLab {
    class MODEL_INFER_API OpenVINOEngine : public BaseEngine {
    private:
        std::shared_ptr<ov::Core> core_;
        std::shared_ptr<ov::Model> model_;
        std::shared_ptr<ov::CompiledModel> compiled_model_;
        std::shared_ptr<ov::InferRequest> infer_request_;

        std::unordered_map<std::string, ioDims> io_dims_map_;

    public:
        OpenVINOEngine(/* args */);
        
        ~OpenVINOEngine() override;

        bool load_dev();

        bool load_model(const std::string& model) override;

        bool infer_model() override;

        void checking() override;

        bool set_dynamic_shapes();

        const std::unordered_map<std::string, ioDims>& get_io_dims() const override;
    };

}; // namespace NexLab


#endif