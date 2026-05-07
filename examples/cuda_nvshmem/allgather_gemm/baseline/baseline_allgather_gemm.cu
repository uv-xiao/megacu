#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cuda_runtime.h>

namespace {

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

__global__ void copy_b_kernel(float const *b, float *gathered_b,
                              ag_gemm_problem problem) {
  auto linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = problem.k * problem.n;
  for (auto index = linear; index < total; index += gridDim.x * blockDim.x) {
    gathered_b[index] = b[index];
  }
}

__global__ void gemm_kernel(float const *a, float const *gathered_b, float *out,
                            ag_gemm_problem problem) {
  auto row = static_cast<std::int64_t>(blockIdx.y);
  auto col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= problem.m || col >= problem.n) {
    return;
  }

  float value = 0.0F;
  for (std::int64_t kk = 0; kk < problem.k; ++kk) {
    value += a[row * problem.k + kk] * gathered_b[kk * problem.n + col];
  }
  out[row * problem.n + col] = value;
}

} // namespace

megacu::status baseline_cuda_nvshmem_allgather_gemm(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, ag_gemm_problem problem) {
  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  auto b_blocks = static_cast<unsigned>((problem.k * problem.n + 127) / 128);
  copy_b_kernel<<<b_blocks, 128, 0, stream>>>(b, gathered_b, problem);
  status = cuda_status(cudaGetLastError(), 11);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  auto c_blocks = static_cast<unsigned>((problem.n + 127) / 128);
  gemm_kernel<<<dim3(c_blocks, problem.m), 128, 0, stream>>>(
      a, gathered_b, out, problem);
  return cuda_status(cudaGetLastError(), 12);
}
