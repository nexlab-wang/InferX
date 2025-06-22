# 基础镜像：使用 NVIDIA CUDA 12.3.2 和 cuDNN 运行时版本
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# 静默安装设置，避免交互式提示
ARG DEBIAN_FRONTEND=noninteractive

# 设置时区
ENV TZ=Asia/Shanghai \
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    git \
    pkg-config \
    ssh \
    pbzip2 \
    bzip2 \
    unzip \
    axel \
    cmake  # 通过 apt 安装 CMake

# 下载并解压 TensorRT 到 /workspace/3dparty 目录
RUN mkdir -p /workspace/3dparty && \
    wget -O /tmp/tensorrt.tar.gz https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.2.0/tars/TensorRT-10.2.0.19.Linux.x86_64-gnu.cuda-12.5.tar.gz  && \
    tar -xvf /tmp/tensorrt.tar.gz -C /workspace/3dparty --transform 's/^TensorRT-10.2.0.19/tensorrt/' && \
    rm -f /tmp/tensorrt.tar.gz

# 下载并解压 OpenCV 到 /workspace/3dparty 目录
RUN mkdir -p /workspace/3dparty && \
    wget -O /tmp/opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip && \
    unzip /tmp/opencv.zip -d /workspace/3dparty && \
    mv /workspace/3dparty/opencv-4.10.0 /workspace/3dparty/opencv_src_code && \
    rm -f /tmp/opencv.zip


#安装openvino
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | tee /etc/apt/sources.list.d/intel-openvino-2024.list && \
    apt-get update && \
    apt-cache search openvino && \
    apt-get install -y openvino-2024.6.0 && \
    rm -f GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB



# 声明容器将使用的端口（例如 8080）
EXPOSE 8080

# 设置工作目录并挂载卷
WORKDIR /workspace
VOLUME /workspace

# 默认启动命令（可选）
CMD ["/bin/bash"]