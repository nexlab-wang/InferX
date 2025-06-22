#ifndef __TO_JSON_HPP__
#define __TO_JSON_HPP__
#include "device_info.hpp"
#include "inferx_config.hpp"
#include "model_param.hpp"
#include "nlohmann/json.hpp"

namespace NexLab {
    // device info
    NLOHMANN_JSON_SERIALIZE_ENUM(DeviceType, {
                                                 {DeviceType::DEV_CPU, "DEV_CPU"},
                                                 {DeviceType::DEV_GPU, "DEV_GPU"},
                                                 {DeviceType::DEV_NPU, "DEV_NPU"},
                                                 {DeviceType::DEV_GPU_CPU, "DEV_GPU_CPU"},
                                                 {DeviceType::DEV_UNKOWN, "DEV_UNKOWN"},
                                             })

    // engine info
    NLOHMANN_JSON_SERIALIZE_ENUM(InferenceEngine, {
                                                      {InferenceEngine::TensorRT, "TensorRT"},
                                                      {InferenceEngine::OpenVINO, "OpenVINO"},
                                                      {InferenceEngine::ONNXRuntime, "ONNXRuntime"},
                                                      {InferenceEngine::LibTorch, "LibTorch"},
                                                      {InferenceEngine::OpenCV, "OpenCV"},
                                                      {InferenceEngine::Unknown, "Unknown"},
                                                  })

    // model info
    NLOHMANN_JSON_SERIALIZE_ENUM(ModelType, {
                                                {ModelType::YOLO_DETECTION, "YOLO_DETECTION"},
                                                {ModelType::YOLO_POSE, "YOLO_POSE"},
                                                {ModelType::YOLO_SEGMENT, "YOLO_SEGMENT"},
                                                {ModelType::YOLO_OBB, "YOLO_OBB"},
                                                {ModelType::RT_DETR, "RT_DETR"},
                                                {ModelType::POINTNET, "POINTNET"},
                                                {ModelType::CUSTOM, "CUSTOM"},
                                            })
    // model params
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
        ModelParamsBase, model_type, batch_size, dynamic_input, model_input_names, model_output_names);

    // image params
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ImageModelParams, model_type, batch_size, dynamic_input, model_input_names,
        model_output_names, num_class, class_names, input_channels, src_h, src_w, dst_h, dst_w, iou_threshold,
        confidence_threshold, num_detection, num_pose, num_mask, mask_size)


    // infex config
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
        InferXConfig, model_path, params_path, dataset_path, infer_engine, dev_type, model_type, enable_visualization);
} // namespace NexLab
#endif