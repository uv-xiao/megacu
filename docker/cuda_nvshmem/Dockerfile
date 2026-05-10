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

RUN useradd --create-home --shell /bin/bash megacu \
  && su megacu -c 'PATH=/home/megacu/.local/bin:${PATH} python3 -m pip install --user --no-cache-dir --no-warn-script-location \
      numpy \
      nvidia-nvshmem-cu12==3.6.5 \
      torch' \
  && mkdir -p /opt/nvidia \
  && NVSHMEM_PY_HOME="$(PYTHONPATH=/home/megacu/.local/lib/python3.10/site-packages python3 -c 'import nvidia, os; print(next(os.path.join(root, "nvshmem") for root in nvidia.__path__ if os.path.isdir(os.path.join(root, "nvshmem"))))')" \
  && ln -s "${NVSHMEM_PY_HOME}" /opt/nvidia/nvshmem

ENV PATH=/home/megacu/.local/bin:${PATH}
ENV PYTHONPATH=/home/megacu/.local/lib/python3.10/site-packages
ENV NVSHMEM_HOME=/opt/nvidia/nvshmem
ENV LD_LIBRARY_PATH=/opt/nvidia/nvshmem/lib:${LD_LIBRARY_PATH}
ENV OMP_NUM_THREADS=1

WORKDIR /workspace/megacu
