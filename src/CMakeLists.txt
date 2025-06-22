cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME inferX)
project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
set(target_name ${PROJECT_NAME})


# 查找CUDA
find_package(CUDAToolkit REQUIRED)

if(CUDA_ARCH_AUTO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    # 自动检测 CUDA 架构
    # 注意：CMake 3.18+ 支持 CMAKE_CUDA_ARCHITECTURES 的 NATIVE 关键字
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set(CMAKE_CUDA_ARCHITECTURES NATIVE)
    else()
        message(WARNING "CMAKE_CUDA_ARCHITECTURES NATIVE requires CMake 3.18 or higher. "
                        "Falling back to manually specified architecture.")
        set(CMAKE_CUDA_ARCHITECTURES 50)  # 默认回退到一个通用架构
    endif()
else()
    # 手动设置 CUDA 架构
    # 替换为你需要的架构编号，例如 50、60、70、80 等
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

# OpenCV
if(WIN32)
    set(OpenCV_DIR ${3rdParty_DIR}/opencv/x64/vc15/lib)
elseif(UNIX)
    set(OpenCV_DIR ${3rdParty_DIR}/opencv/lib/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)

# TensorRT
if(WIN32)
    set(TensorRT_INCLUDE_DIR ${3rdParty_DIR}/tensorrt/include)
    set(TensorRT_LIBRARY_DIR ${3rdParty_DIR}/tensorrt/lib)
elseif(UNIX)
    set(TensorRT_INCLUDE_DIR ${3rdParty_DIR}/tensorrt/include)
    set(TensorRT_LIBRARY_DIR ${3rdParty_DIR}/tensorrt/lib)
endif()

# OpenVINO
set(OpenVINO_DIR ${3rdParty_DIR}/openvino/runtime/cmake)
find_package(OpenVINO REQUIRED)
if (OpenVINO_FOUND)
    message(STATUS "OpenVINO found: ${OpenVINO_VERSION}")
else()
    message(FATAL_ERROR "OpenVINO not found!")
endif()

# 包含目录
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
)

# 链接目录
link_directories(
    ${CUDA_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
    ${TensorRT_LIBRARY_DIR}
)
message("TensorRT_LIBRARY_DIR: ${TensorRT_LIBRARY_DIR}")

# 源文件
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ./*.cpp ./*.cu ../include/*.h ../include/*.hpp)

# 添加库
add_library(${PROJECT_NAME} SHARED ${sources})
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/utf-8">)

# 设置动态库导出符号（Windows）
if(WIN32)
    target_compile_definitions(${target_name} PRIVATE -DMODEL_INFER_EXPORT)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)  # 自动导出所有符号
endif()

# 包含目录
target_include_directories(${target_name} PUBLIC ./ ../include)
target_include_directories(${PROJECT_NAME} PRIVATE 
                            ${CUDA_INCLUDE_DIRS} 
                            ${OpenCV_INCLUDE_DIRS} 
                            ${TensorRT_INCLUDE_DIR}
                            ${OpenVINO_INCLUDE_DIRS})

# 链接库
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
                            ${CUDA_LIBRARIES} 
                            ${OpenCV_LIBS} 
                            nvinfer_10 
                            nvonnxparser_10
                            CUDA::cudart
                            openvino::runtime)
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
                            ${CUDA_LIBRARIES} 
                            ${OpenCV_LIBS} 
                            nvinfer
                            nvonnxparser
                            CUDA::cudart
                            openvino::runtime)
endif()


# 编译定义
# target_compile_definitions(${target_name} PRIVATE
#     ${CUDA_DEFINITIONS}
# )


# # 编译选项
# target_compile_options(${target_name} PRIVATE
#     ${ARCH_FLAGS}
# )

# 启用CUDA分离编译（可选）
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
