#pragma once

#include <cuda_runtime.h>

namespace megacu::platform::cuda {

struct device_context {
  __device__ int block_id() const { return blockIdx.x; }

  __device__ int grid_blocks() const { return gridDim.x; }

  __device__ int thread_id() const { return threadIdx.x; }

  __device__ int block_threads() const { return blockDim.x; }

  __device__ void sync_block() const { __syncthreads(); }
};

struct launch_config {
  int blocks = 1;
  int threads = 1;
};

template <class Program> __global__ void megakernel_entry(Program program) {
  program(device_context{});
}

template <class Program>
cudaError_t launch_megakernel(cudaStream_t stream, launch_config config,
                              Program program) {
  megakernel_entry<<<config.blocks, config.threads, 0, stream>>>(program);
  return cudaGetLastError();
}

} // namespace megacu::platform::cuda
