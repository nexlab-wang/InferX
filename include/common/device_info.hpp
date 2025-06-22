#ifndef __DEVICE_INFO_HPP__
#define __DEVICE_INFO_HPP__
#include "dllexport.hpp"

#include <iostream>
namespace NexLab {
    enum DeviceType { DEV_CPU, DEV_GPU, DEV_NPU, DEV_GPU_CPU, DEV_UNKOWN };

    class MODEL_INFER_API DeviceInfo {
    public:
        DeviceInfo();
        DeviceInfo(DeviceType dev_type, uint8_t dev_id);

        void set_dev_type(DeviceType dev_type);
        DeviceType get_dev_type() const;

        void set_dev_id(uint8_t dev_id);
        uint8_t get_dev_id() const;

    private:
        DeviceType dev_type_;
        uint8_t dev_id_;
    };
} // namespace NexLab
#endif