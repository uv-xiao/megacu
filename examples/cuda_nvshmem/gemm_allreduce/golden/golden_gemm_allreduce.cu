#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;

golden_status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {golden_status_code::launch_error, detail, cudaGetErrorString(error)};
}

constexpr std::int64_t ceil_div(std::int64_t value, std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

golden_status validate(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.tile_m <= 0 || problem.tile_n <= 0 ||
      problem.compute_ctas <= 0 || problem.comm_ctas <= 0) {
    return {golden_status_code::invalid_argument, 1, "invalid problem"};
  }
  if (workspace.a.data == nullptr || workspace.b.data == nullptr ||
      workspace.partial.data == nullptr || workspace.c.data == nullptr ||
      workspace.barriers == nullptr || launch.stream == nullptr) {
    return {golden_status_code::invalid_argument, 2, "missing storage"};
  }

  auto elements = problem.m * problem.n;
  auto a_bytes = problem.m * problem.k * static_cast<std::int64_t>(sizeof(float));
  auto b_bytes = problem.k * problem.n * static_cast<std::int64_t>(sizeof(float));
  auto c_bytes = elements * static_cast<std::int64_t>(sizeof(float));
  auto tiles = ceil_div(problem.m, problem.tile_m) *
               ceil_div(problem.n, problem.tile_n);
  auto barrier_bytes = tiles * static_cast<std::int64_t>(sizeof(int));
  if (workspace.a.bytes < a_bytes || workspace.b.bytes < b_bytes ||
      workspace.c.bytes < c_bytes || workspace.partial.bytes < 2 * c_bytes ||
      workspace.barrier_bytes < barrier_bytes) {
    return {golden_status_code::invalid_argument, 3, "storage too small"};
  }
  return {};
}

__device__ std::int64_t device_ceil_div(std::int64_t value, std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

__device__ void compute_tile(
    float const *a,
    float const *b,
    float volatile *partial,
    std::int64_t tile_id,
    std::int64_t m,
    std::int64_t n,
    std::int64_t k,
    std::int32_t tile_m,
    std::int32_t tile_n) {
  auto n_tiles = device_ceil_div(n, tile_n);
  auto tile_row = tile_id / n_tiles;
  auto tile_col = tile_id % n_tiles;
  auto row_begin = tile_row * tile_m;
  auto col_begin = tile_col * tile_n;
  auto tile_elements = static_cast<std::int64_t>(tile_m) * tile_n;

  for (auto offset = static_cast<std::int64_t>(threadIdx.x);
       offset < tile_elements;
       offset += blockDim.x) {
    auto local_row = offset / tile_n;
    auto local_col = offset % tile_n;
    auto row = row_begin + local_row;
    auto col = col_begin + local_col;
    if (row >= m || col >= n) {
      continue;
    }
    float value = 0.0f;
    for (std::int64_t kk = 0; kk < k; ++kk) {
      value += a[row * k + kk] * b[kk * n + col];
    }
    partial[row * n + col] = value;
  }
}

__device__ void copy_tile(
    float volatile const *partial,
    float *out,
    std::int64_t tile_id,
    std::int64_t m,
    std::int64_t n,
    std::int32_t tile_m,
    std::int32_t tile_n) {
  auto n_tiles = device_ceil_div(n, tile_n);
  auto tile_row = tile_id / n_tiles;
  auto tile_col = tile_id % n_tiles;
  auto row_begin = tile_row * tile_m;
  auto col_begin = tile_col * tile_n;
  auto tile_elements = static_cast<std::int64_t>(tile_m) * tile_n;

  for (auto offset = static_cast<std::int64_t>(threadIdx.x);
       offset < tile_elements;
       offset += blockDim.x) {
    auto local_row = offset / tile_n;
    auto local_col = offset % tile_n;
    auto row = row_begin + local_row;
    auto col = col_begin + local_col;
    if (row >= m || col >= n) {
      continue;
    }
    out[row * n + col] = partial[row * n + col];
  }
}

__global__ void persistent_gemm_notify_kernel(
    float const *a,
    float const *b,
    float volatile *partial,
    int *barriers,
    std::int64_t m,
    std::int64_t n,
    std::int64_t k,
    std::int32_t tile_m,
    std::int32_t tile_n,
    std::int64_t tiles) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
       tile_id < tiles;
       tile_id += gridDim.x) {
    compute_tile(a, b, partial, tile_id, m, n, k, tile_m, tile_n);
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicExch(barriers + tile_id, static_cast<int>(tile_id + 1));
    }
    __syncthreads();
  }
}

__global__ void wait_copy_tiles_kernel(
    float volatile const *partial,
    float *out,
    int *barriers,
    std::int64_t m,
    std::int64_t n,
    std::int32_t tile_m,
    std::int32_t tile_n,
    std::int64_t tiles) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
       tile_id < tiles;
       tile_id += gridDim.x) {
    auto ready_value = static_cast<int>(tile_id + 1);
    while (atomicAdd(barriers + tile_id, 0) != ready_value) {
    }
    copy_tile(partial, out, tile_id, m, n, tile_m, tile_n);
  }
}

__global__ void fused_overlap_kernel(
    float const *a,
    float const *b,
    float volatile *partial,
    float *out,
    int *barriers,
    std::int64_t m,
    std::int64_t n,
    std::int64_t k,
    std::int32_t tile_m,
    std::int32_t tile_n,
    std::int64_t tiles,
    std::int32_t comm_ctas) {
  if (blockIdx.x < comm_ctas) {
    for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
         tile_id < tiles;
         tile_id += comm_ctas) {
      auto ready_value = static_cast<int>(tile_id + 1);
      while (atomicAdd(barriers + tile_id, 0) != ready_value) {
      }
      copy_tile(partial, out, tile_id, m, n, tile_m, tile_n);
    }
    return;
  }

  auto compute_ctas = gridDim.x - comm_ctas;
  auto compute_id = blockIdx.x - comm_ctas;
  for (auto tile_id = static_cast<std::int64_t>(compute_id);
       tile_id < tiles;
       tile_id += compute_ctas) {
    compute_tile(a, b, partial, tile_id, m, n, k, tile_m, tile_n);
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicExch(barriers + tile_id, static_cast<int>(tile_id + 1));
    }
    __syncthreads();
  }
}

__global__ void copy_linear_kernel(
    float const *src,
    float *dst,
    std::int64_t elements) {
  auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    dst[index] = src[index];
  }
}

golden_status copy_reduced_to_output(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  auto stream = static_cast<cudaStream_t>(launch.stream);
  auto elements = problem.m * problem.n;
  auto *partial = static_cast<float *>(workspace.partial.data);
  auto *reduced = partial + elements;
  auto blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  copy_linear_kernel<<<blocks, kThreads, 0, stream>>>(
      reduced,
      static_cast<float *>(workspace.c.data),
      elements);
  auto status = cuda_status(cudaGetLastError(), 40);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return cuda_status(cudaStreamSynchronize(stream), 41);
}

golden_status launch_phased_local(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  auto status = validate(workspace, launch, problem);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  status = cuda_status(cudaSetDevice(launch.device_ordinal), 10);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(launch.stream);
  auto tiles = ceil_div(problem.m, problem.tile_m) *
               ceil_div(problem.n, problem.tile_n);
  auto *barriers = static_cast<int *>(workspace.barriers);
  status = cuda_status(
      cudaMemsetAsync(barriers, 0, tiles * sizeof(int), stream), 11);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  cudaStream_t comm_stream = nullptr;
  cudaEvent_t ready = nullptr;
  status = cuda_status(cudaStreamCreate(&comm_stream), 12);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  status = cuda_status(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), 13);
  if (status.code != golden_status_code::ok) {
    cudaStreamDestroy(comm_stream);
    return status;
  }
  cudaEventRecord(ready, stream);
  cudaStreamWaitEvent(comm_stream, ready, 0);

  persistent_gemm_notify_kernel<<<problem.compute_ctas, kThreads, 0, stream>>>(
      static_cast<float const *>(workspace.a.data),
      static_cast<float const *>(workspace.b.data),
      static_cast<float *>(workspace.partial.data),
      barriers,
      problem.m,
      problem.n,
      problem.k,
      problem.tile_m,
      problem.tile_n,
      tiles);
  status = cuda_status(cudaGetLastError(), 14);
  if (status.code == golden_status_code::ok) {
    wait_copy_tiles_kernel<<<problem.comm_ctas, kThreads, 0, comm_stream>>>(
        static_cast<float const *>(workspace.partial.data),
        static_cast<float *>(workspace.c.data),
        barriers,
        problem.m,
        problem.n,
        problem.tile_m,
        problem.tile_n,
        tiles);
    status = cuda_status(cudaGetLastError(), 15);
  }
  if (status.code == golden_status_code::ok) {
    status = cuda_status(cudaStreamWaitEvent(stream, ready, 0), 16);
  }
  if (status.code == golden_status_code::ok) {
    status = cuda_status(cudaStreamSynchronize(comm_stream), 17);
  }
  cudaEventDestroy(ready);
  cudaStreamDestroy(comm_stream);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return cuda_status(cudaStreamSynchronize(stream), 18);
}

golden_status launch_overlap_local(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  auto status = validate(workspace, launch, problem);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  status = cuda_status(cudaSetDevice(launch.device_ordinal), 20);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(launch.stream);
  auto tiles = ceil_div(problem.m, problem.tile_m) *
               ceil_div(problem.n, problem.tile_n);
  auto *barriers = static_cast<int *>(workspace.barriers);
  status = cuda_status(
      cudaMemsetAsync(barriers, 0, tiles * sizeof(int), stream), 21);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  auto blocks = problem.comm_ctas + problem.compute_ctas;
  fused_overlap_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<float const *>(workspace.a.data),
      static_cast<float const *>(workspace.b.data),
      static_cast<float *>(workspace.partial.data),
      static_cast<float *>(workspace.c.data),
      barriers,
      problem.m,
      problem.n,
      problem.k,
      problem.tile_m,
      problem.tile_n,
      tiles,
      problem.comm_ctas);
  status = cuda_status(cudaGetLastError(), 22);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return cuda_status(cudaStreamSynchronize(stream), 23);
}

golden_status run_multi_card_reduce(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem,
    golden_gemm_ar_comm_ops const *ops) {
  if (team.team_n_pes <= 1) {
    return {};
  }
  if (ops == nullptr || ops->sum_reduce_f32 == nullptr) {
    return {golden_status_code::invalid_argument, 30, "missing reduction op"};
  }
  auto elements = problem.m * problem.n;
  auto *partial = static_cast<float *>(workspace.partial.data);
  auto *reduced = partial + elements;
  auto status = ops->sum_reduce_f32(
      team.team,
      reduced,
      partial,
      elements,
      launch.stream);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return copy_reduced_to_output(workspace, launch, problem);
}

}  // namespace

golden_status golden_phased_single_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  return launch_phased_local(workspace, launch, problem);
}

golden_status golden_overlap_single_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem) {
  return launch_overlap_local(workspace, launch, problem);
}

golden_status golden_phased_multi_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem,
    golden_gemm_ar_comm_ops const *ops) {
  auto status = launch_phased_local(workspace, launch, problem);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return run_multi_card_reduce(workspace, launch, team, problem, ops);
}

golden_status golden_overlap_multi_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem,
    golden_gemm_ar_comm_ops const *ops) {
  auto status = launch_overlap_local(workspace, launch, problem);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return run_multi_card_reduce(workspace, launch, team, problem, ops);
}
