#ifndef __MODEL_PARAM_HPP__
#define __MODEL_PARAM_HPP__

#include <stdexcept>
#include <string>
#include <vector>


namespace NexLab {

    enum class ModelType { YOLO_DETECTION, YOLO_POSE, YOLO_SEGMENT, YOLO_OBB, RT_DETR, POINTNET, CUSTOM };
    
    // Abstract base class for model inference parameters
    class ModelParamsBase {
    public:
        virtual ~ModelParamsBase() = default;

        ModelType model_type;

        size_t batch_size{1}; // Batch size
        bool dynamic_input{true}; // Supports dynamic input sizes
        std::vector<std::string> model_input_names{};
        std::vector<std::string> model_output_names{};

        virtual void validate() const {
            if (batch_size <= 0) {

                throw std::invalid_argument("Batch size must be greater than zero.");
            }
        }
    };

    // Parameters for image-based models (e.g., YOLO, ViT, DETR)
    class ImageModelParams : public ModelParamsBase {
    public:
        int num_class{80}; // coco
        std::vector<std::string> class_names;
        int input_channels{3}; // Number of input channels (e.g., RGB = 3)
        int src_h{0}, src_w{0}; // Original image dimensions  图像的尺寸应该也删除掉
        int dst_h{640}, dst_w{640}; // Model input dimensions

        float iou_threshold{0.5f}; // IOU threshold for object detection
        float confidence_threshold{0.5f}; // Confidence threshold for predictions

        int num_detection{8400}; // Max number of detections
        int num_pose{0}; // Number of pose
        int num_mask{0};
        int mask_size{0};

        void validate() const override {

            ModelParamsBase::validate();

            if (dst_h <= 0 || dst_w <= 0) {

                throw std::invalid_argument("Input dimensions must be positive.");
            }
        }
    };

    // Parameters for point cloud-based models (e.g., PointNet, PointNet++)
    class PointCloudModelParams : public ModelParamsBase {
    public:
        size_t num_points{1024}; // Number of input points
        bool use_xyz{true}; // Include XYZ coordinates

        void validate() const override {

            ModelParamsBase::validate();

            if (num_points == 0) {

                throw std::invalid_argument("Number of points must be greater than zero.");
            }
        }
    };
} // namespace NexLab

#endif