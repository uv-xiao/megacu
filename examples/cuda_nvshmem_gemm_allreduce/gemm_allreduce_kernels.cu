#include "examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void gemm_partial_kernel(
    float const *a,
    float const *b,
    float *partial,
    std::int64_t m,
    std::int64_t n,
    std::int64_t k) {
  auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto elements = m * n;
  if (index >= elements) {
    return;
  }

  auto row = index / n;
  auto col = index % n;
  float value = 0.0f;
  for (std::int64_t kk = 0; kk < k; ++kk) {
    value += a[row * k + kk] * b[kk * n + col];
  }
  partial[index] = value;
}

__global__ void write_reduced_kernel(
    float const *reduced,
    float *out,
    std::int64_t elements) {
  auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    out[index] = reduced[index];
  }
}

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

}  // namespace

extern "C" megacu::status megacu_cuda_gemm_allreduce_f32(
    gemm_ar_workspace workspace,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    gemm_ar_comm_ops const *ops) {
  auto device_status = cuda_status(cudaSetDevice(launch.device_ordinal), 1);
  if (device_status.code != megacu::status_code::ok) {
    return device_status;
  }

  auto stream = static_cast<cudaStream_t>(launch.stream);
  auto elements = problem.m * problem.n;
  auto *partial = static_cast<float *>(workspace.partial.buffer.data);
  auto *reduced = partial + elements;

  constexpr int kThreads = 128;
  auto blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  gemm_partial_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<float const *>(workspace.a.data),
      static_cast<float const *>(workspace.b.data),
      partial,
      problem.m,
      problem.n,
      problem.k);
  auto launch_status = cuda_status(cudaGetLastError(), 2);
  if (launch_status.code != megacu::status_code::ok) {
    return launch_status;
  }

  auto comm_status =
      ops->sum_reduce_f32(team, reduced, partial, elements, launch.stream);
  if (comm_status.code != megacu::status_code::ok) {
    return comm_status;
  }

  write_reduced_kernel<<<blocks, kThreads, 0, stream>>>(
      reduced,
      static_cast<float *>(workspace.c.data),
      elements);
  launch_status = cuda_status(cudaGetLastError(), 3);
  if (launch_status.code != megacu::status_code::ok) {
    return launch_status;
  }

  return cuda_status(cudaStreamSynchronize(stream), 4);
}
