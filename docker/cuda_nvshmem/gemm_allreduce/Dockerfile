FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG NVSHMEM_PACKAGE=libnvshmem3-dev-cuda-12
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libopenmpi-dev \
    ninja-build \
    openmpi-bin \
    ${NVSHMEM_PACKAGE} \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/megacu
