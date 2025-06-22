#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>


namespace NexLab {
    namespace utils {
        /// @brief lode model file
        /// @param file-model file path
        /// @return
        std::vector<unsigned char> inline loadModel(const std::string& file) {
            std::ifstream in(file, std::ios::in | std::ios::binary);

            if (!in.is_open()) {

                in.close();
                return {};
            }

            in.seekg(0, std::ios::end);
            size_t length = in.tellg();

            std::vector<uint8_t> data;
            if (length > 0) {

                in.seekg(0, std::ios::beg);
                data.resize(length);
                in.read((char*) &data[0], length);
            }

            in.close();
            return data;
        }


        inline std::vector<std::string> load_dataset_list(const std::string& dataset_path) {
            if (!std::filesystem::exists(dataset_path) || !std::filesystem::is_directory(dataset_path)) {
                std::cerr << "Error: Dataset path is error!\n";
                return {};
            }

            std::vector<std::string> file_list;

            const std::vector<std::string> supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
                ".pcd", ".ply", ".las", ".laz", ".xyz", ".pts", ".csv"};

            for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(
                        ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

                    if (std::find(supported_extensions.begin(), supported_extensions.end(), ext)
                        != supported_extensions.end()) {
                        file_list.emplace_back((dataset_path / entry.path().filename()).string());
                    }
                }
            }

            return file_list;
        }
    } // namespace utils
} // namespace NexLab

#endif