#include "examples/cuda_nvshmem/gemm_reduce_scatter/common/gemm_reduce_scatter.h"

#include <cuda_runtime.h>

namespace {

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

__global__ void baseline_gemm_reduce_scatter_kernel(
    float const *a, float const *b, float *partial, float *out,
    gemm_rs_problem problem, int my_pe, int n_pes) {
  auto row = static_cast<std::int64_t>(blockIdx.y);
  auto col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= problem.m || col >= problem.n) {
    return;
  }

  float value = 0.0F;
  for (std::int64_t kk = 0; kk < problem.k; ++kk) {
    value += a[row * problem.k + kk] * b[kk * problem.n + col];
  }

  auto index = row * problem.n + col;
  partial[index] = value;
  out[index] = row % n_pes == my_pe ? value : 0.0F;
}

} // namespace

megacu::status baseline_cuda_nvshmem_gemm_reduce_scatter(
    gemm_rs_driver driver, float const *a, float const *b, float *partial,
    float *out, gemm_rs_problem problem) {
  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  auto blocks_x = static_cast<unsigned>((problem.n + 127) / 128);
  baseline_gemm_reduce_scatter_kernel<<<dim3(blocks_x, problem.m), 128, 0,
                                        stream>>>(
      a, b, partial, out, problem, driver.team.team_my_pe,
      driver.team.team_n_pes);
  return cuda_status(cudaGetLastError(), 11);
}
