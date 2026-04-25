#include "examples/cuda_nvshmem/common/gemm_allreduce/golden/golden_gemm_allreduce.h"

#include <cuda_runtime.h>

#include <cstdint>

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvshmem.h>

extern "C" void nvshmem_barrier_all();
#endif

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
    std::int64_t tiles,
    int ready_base) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
       tile_id < tiles;
       tile_id += gridDim.x) {
    compute_tile(a, b, partial, tile_id, m, n, k, tile_m, tile_n);
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicExch(barriers + tile_id, ready_base + static_cast<int>(tile_id + 1));
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
    std::int64_t tiles,
    int ready_base) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
       tile_id < tiles;
       tile_id += gridDim.x) {
    auto ready_value = ready_base + static_cast<int>(tile_id + 1);
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
    std::int32_t comm_ctas,
    int ready_base) {
  if (blockIdx.x < comm_ctas) {
    for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
         tile_id < tiles;
         tile_id += comm_ctas) {
      auto ready_value = ready_base + static_cast<int>(tile_id + 1);
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
      atomicExch(barriers + tile_id, ready_base + static_cast<int>(tile_id + 1));
    }
    __syncthreads();
  }
}

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
__device__ void wait_tile_ready(
    int const *barriers,
    std::int64_t tile_id,
    int ready_value,
    int pe,
    int my_pe) {
  if (pe == my_pe) {
    while (atomicAdd(const_cast<int *>(barriers + tile_id), 0) != ready_value) {
    }
    return;
  }

  while (nvshmem_int_g(barriers + tile_id, pe) != ready_value) {
  }
}

__device__ void allreduce_tile_from_symmetric_partials(
    float volatile const *partial,
    float *out,
    int const *barriers,
    std::int64_t tile_id,
    std::int64_t m,
    std::int64_t n,
    std::int32_t tile_m,
    std::int32_t tile_n,
    int my_pe,
    int n_pes,
    int ready_base) {
  auto ready_value = ready_base + static_cast<int>(tile_id + 1);
  for (int pe = 0; pe < n_pes; ++pe) {
    wait_tile_ready(barriers, tile_id, ready_value, pe, my_pe);
  }

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

    auto index = row * n + col;
    float value = 0.0f;
    for (int pe = 0; pe < n_pes; ++pe) {
      if (pe == my_pe) {
        value += partial[index];
      } else {
        value += nvshmem_float_g(
            const_cast<float const *>(partial + index),
            pe);
      }
    }
    out[index] = value;
  }
}

__global__ void phased_nvshmem_allreduce_kernel(
    float volatile const *partial,
    float *out,
    int const *barriers,
    std::int64_t m,
    std::int64_t n,
    std::int32_t tile_m,
    std::int32_t tile_n,
    std::int64_t tiles,
    int my_pe,
    int n_pes,
    int ready_base) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x);
       tile_id < tiles;
       tile_id += gridDim.x) {
    allreduce_tile_from_symmetric_partials(
        partial,
        out,
        barriers,
        tile_id,
        m,
        n,
        tile_m,
        tile_n,
        my_pe,
        n_pes,
        ready_base);
  }
}

#endif

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
      tiles,
      0);
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
        tiles,
        0);
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
      problem.comm_ctas,
      0);
  status = cuda_status(cudaGetLastError(), 22);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  return cuda_status(cudaStreamSynchronize(stream), 23);
}

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
golden_status enter_collective_after_local_reset(
    cudaStream_t stream,
    std::uint16_t detail) {
  auto status = cuda_status(cudaStreamSynchronize(stream), detail);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  nvshmem_barrier_all();
  return {};
}

golden_status launch_phased_multi_card_device(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem) {
  auto status = validate(workspace, launch, problem);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  if (team.team_n_pes <= 1 || team.team_my_pe < 0 ||
      team.team_my_pe >= team.team_n_pes) {
    return {golden_status_code::invalid_argument, 30, "invalid team"};
  }
  status = cuda_status(cudaSetDevice(launch.device_ordinal), 31);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(launch.stream);
  auto tiles = ceil_div(problem.m, problem.tile_m) *
               ceil_div(problem.n, problem.tile_n);
  static int collective_epoch = 0;
  auto ready_base = (++collective_epoch) * static_cast<int>(tiles + 1);
  auto *barriers = static_cast<int *>(workspace.barriers);
  status = cuda_status(
      cudaMemsetAsync(barriers, 0, tiles * sizeof(int), stream), 32);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  status = enter_collective_after_local_reset(stream, 33);
  if (status.code != golden_status_code::ok) {
    return status;
  }

  cudaStream_t comm_stream = nullptr;
  cudaEvent_t ready = nullptr;
  status = cuda_status(cudaStreamCreate(&comm_stream), 34);
  if (status.code != golden_status_code::ok) {
    return status;
  }
  status = cuda_status(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), 35);
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
      tiles,
      ready_base);
  status = cuda_status(cudaGetLastError(), 36);
  if (status.code == golden_status_code::ok) {
    phased_nvshmem_allreduce_kernel<<<problem.comm_ctas, kThreads, 0, comm_stream>>>(
        static_cast<float const *>(workspace.partial.data),
        static_cast<float *>(workspace.c.data),
        barriers,
        problem.m,
        problem.n,
        problem.tile_m,
        problem.tile_n,
        tiles,
        team.team_my_pe,
        team.team_n_pes,
        ready_base);
    status = cuda_status(cudaGetLastError(), 37);
  }
  if (status.code == golden_status_code::ok) {
    status = cuda_status(cudaStreamSynchronize(comm_stream), 38);
  }
  if (status.code == golden_status_code::ok) {
    status = cuda_status(cudaStreamSynchronize(stream), 39);
  }
  cudaEventDestroy(ready);
  cudaStreamDestroy(comm_stream);
  if (status.code == golden_status_code::ok) {
    nvshmem_barrier_all();
  }
  return status;
}

golden_status launch_overlap_multi_card_device(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem) {
  return launch_phased_multi_card_device(workspace, launch, team, problem);
}
#endif

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
    golden_gemm_ar_problem problem) {
#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
  return launch_phased_multi_card_device(workspace, launch, team, problem);
#else
  (void)workspace;
  (void)launch;
  (void)team;
  (void)problem;
  return {
      golden_status_code::unsupported,
      60,
      "device-side NVSHMEM path was not built"};
#endif
}

golden_status golden_overlap_multi_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem) {
#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
  return launch_overlap_multi_card_device(workspace, launch, team, problem);
#else
  (void)workspace;
  (void)launch;
  (void)team;
  (void)problem;
  return {
      golden_status_code::unsupported,
      61,
      "device-side NVSHMEM path was not built"};
#endif
}
