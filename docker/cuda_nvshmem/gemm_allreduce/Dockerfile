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
    python3 \
    python3-pip \
    ${NVSHMEM_PACKAGE} \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir nvidia-nvshmem-cu12==3.6.5 \
  && mkdir -p /opt/nvidia \
  && NVSHMEM_PY_HOME="$(python3 -c 'import nvidia, os; print(next(os.path.join(root, "nvshmem") for root in nvidia.__path__ if os.path.isdir(os.path.join(root, "nvshmem"))))')" \
  && ln -s "${NVSHMEM_PY_HOME}" /opt/nvidia/nvshmem

ENV NVSHMEM_HOME=/opt/nvidia/nvshmem
ENV LD_LIBRARY_PATH=/opt/nvidia/nvshmem/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/megacu
