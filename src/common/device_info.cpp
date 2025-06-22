#include "common/device_info.hpp"

namespace NexLab {
    DeviceInfo::DeviceInfo() : dev_type_(DEV_CPU), dev_id_(0) {}

    DeviceInfo::DeviceInfo(DeviceType dev_type, uint8_t dev_id) : dev_type_(dev_type), dev_id_(dev_id) {}

    void DeviceInfo::set_dev_type(DeviceType dev_type) {
        dev_type_ = dev_type;
    }

    DeviceType DeviceInfo::get_dev_type() const {
        return dev_type_;
    }

    void DeviceInfo::set_dev_id(uint8_t dev_id) {
        dev_id_ = dev_id;
    }

    uint8_t DeviceInfo::get_dev_id() const {
        return dev_id_;
    }
} // namespace NexLab