#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cuda_runtime.h>

namespace {

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

__global__ void golden_allgather_gemm_kernel(
    float const *a, float const *b, float *gathered_b, float *out,
    ag_gemm_problem problem) {
  auto linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto b_total = problem.k * problem.n;
  for (auto index = linear; index < b_total; index += gridDim.x * blockDim.x) {
    gathered_b[index] = b[index];
  }

  auto c_total = problem.m * problem.n;
  for (auto index = linear; index < c_total; index += gridDim.x * blockDim.x) {
    auto row = index / problem.n;
    auto col = index % problem.n;

    float value = 0.0F;
    for (std::int64_t kk = 0; kk < problem.k; ++kk) {
      value += a[row * problem.k + kk] * gathered_b[kk * problem.n + col];
    }
    out[index] = value;
  }
}

} // namespace

megacu::status golden_cuda_nvshmem_allgather_gemm(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, ag_gemm_problem problem) {
  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 1);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  golden_allgather_gemm_kernel<<<2, 128, 0, stream>>>(a, b, gathered_b, out,
                                                      problem);
  return cuda_status(cudaGetLastError(), 2);
}
