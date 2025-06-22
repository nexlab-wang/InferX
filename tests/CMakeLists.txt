cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME test)

set(target_name ${PROJECT_NAME})

# 查找CUDA
find_package(CUDAToolkit REQUIRED)

if(CUDA_ARCH_AUTO)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set(CMAKE_CUDA_ARCHITECTURES NATIVE)
    else()
        message(WARNING "CMAKE_CUDA_ARCHITECTURES NATIVE requires CMake 3.18 or higher. "
                        "Falling back to manually specified architecture.")
        set(CMAKE_CUDA_ARCHITECTURES 50) 
    endif()
else()
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



file(GLOB_RECURSE sources CONFIGURE_DEPENDS 
    ../src/*.cpp 
    ../include/*.h 
    ../include/*.hpp 
    ./*.cpp 
    ./*.hpp 
    ./*.h
)

message(STATUS "Found sources:")
foreach(src ${sources})
    message(STATUS "    ${src}")  
endforeach()


# add_executable(${target_name} test_yolo.cpp)
add_executable(${target_name} test_rt-detr.cpp)

target_include_directories(${target_name} PUBLIC ./)
target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR} ${OpenVINO_INCLUDE_DIRS})

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
                            ${CUDA_LIBRARIES} 
                            ${OpenCV_LIBS} 
                            nvinfer_10 
                            nvonnxparser_10
                            cudart
                            openvino::runtime)
    target_link_libraries(${target_name} PRIVATE inferX)
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
                            ${CUDA_LIBRARIES} 
                            ${OpenCV_LIBS} 
                            nvinfer 
                            nvonnxparser
                            CUDA::cudart
                            openvino::runtime)
    target_link_libraries(${target_name} PRIVATE inferX)
endif()

get_target_property(LINK_LIBRARIES ${target_name} LINK_LIBRARIES)
message(STATUS "Linked libraries for ${target_name}:")
foreach(lib ${LINK_LIBRARIES})
    message(STATUS "    ${lib}")
endforeach()