#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

#include <cuda_runtime.h>

#include <cstdint>

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
#include <nvshmem.h>

extern "C" void nvshmem_barrier_all();
#endif

namespace {

constexpr int kThreads = 128;

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

constexpr std::int64_t ceil_div(std::int64_t value, std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

megacu::status validate_problem(gemm_ar_driver driver, float const *a,
                                float const *b, float *partial, float *out,
                                void *events, gemm_ar_problem problem) {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.tile_m <= 0 || problem.tile_n <= 0) {
    return {megacu::status_code::invalid_argument, 1, "invalid problem"};
  }
  if (a == nullptr || b == nullptr || partial == nullptr || out == nullptr ||
      events == nullptr) {
    return {megacu::status_code::invalid_argument, 2, "missing storage"};
  }
  if (driver.launch.stream == nullptr) {
    return {};
  }
  return {};
}

__device__ std::int64_t device_ceil_div(std::int64_t value,
                                        std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

__device__ void manual_compute_tile(float const *a, float const *b,
                                    float volatile *partial,
                                    std::int64_t tile_id, std::int64_t m,
                                    std::int64_t n, std::int64_t k,
                                    std::int32_t tile_m, std::int32_t tile_n) {
  auto n_tiles = device_ceil_div(n, tile_n);
  auto tile_row = tile_id / n_tiles;
  auto tile_col = tile_id % n_tiles;
  auto row_begin = tile_row * tile_m;
  auto col_begin = tile_col * tile_n;
  auto tile_elements = static_cast<std::int64_t>(tile_m) * tile_n;

  for (auto offset = static_cast<std::int64_t>(threadIdx.x);
       offset < tile_elements; offset += blockDim.x) {
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

__device__ void manual_publish_tile(int *events, std::int64_t tile_id,
                                    std::int64_t tiles, int my_pe, int n_pes,
                                    int ready_value) {
  megacu::cuda::device::fence_system();
  auto slot = static_cast<std::int64_t>(my_pe) * tiles + tile_id;
  for (int pe = 0; pe < n_pes; ++pe) {
    if (pe == my_pe) {
      megacu::cuda::device::signal_ready(events, slot, ready_value);
    } else {
#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
      nvshmem_int_p(events + slot, ready_value, pe);
#endif
    }
  }
}

__device__ void manual_acquire_tile(int const *events, std::int64_t tile_id,
                                    std::int64_t tiles, int n_pes) {
  auto ready_value = static_cast<int>(tile_id + 1);
  for (int pe = 0; pe < n_pes; ++pe) {
    auto slot = static_cast<std::int64_t>(pe) * tiles + tile_id;
    megacu::cuda::device::wait_ready(events, slot, ready_value);
  }
}

__device__ void manual_reduce_tile(float volatile const *partial, float *out,
                                   std::int64_t tile_id, std::int64_t m,
                                   std::int64_t n, std::int32_t tile_m,
                                   std::int32_t tile_n, int my_pe, int n_pes) {
  auto n_tiles = device_ceil_div(n, tile_n);
  auto tile_row = tile_id / n_tiles;
  auto tile_col = tile_id % n_tiles;
  auto row_begin = tile_row * tile_m;
  auto col_begin = tile_col * tile_n;
  auto tile_elements = static_cast<std::int64_t>(tile_m) * tile_n;

  for (auto offset = static_cast<std::int64_t>(threadIdx.x);
       offset < tile_elements; offset += blockDim.x) {
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
#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
        value += megacu::nvshmem::device::remote_float(
            const_cast<float const *>(partial), index, pe);
#endif
      }
    }
    out[index] = value;
  }
}

__global__ void manual_gemm_allreduce_megakernel(
    float const *a, float const *b, float volatile *partial, float *out,
    int *events, std::int64_t m, std::int64_t n, std::int64_t k,
    std::int32_t tile_m, std::int32_t tile_n, std::int64_t tiles, int my_pe,
    int n_pes) {
  for (auto tile_id = static_cast<std::int64_t>(blockIdx.x); tile_id < tiles;
       tile_id += gridDim.x) {
    manual_compute_tile(a, b, partial, tile_id, m, n, k, tile_m, tile_n);
    __syncthreads();
    if (threadIdx.x == 0) {
      manual_publish_tile(events, tile_id, tiles, my_pe, n_pes,
                          static_cast<int>(tile_id + 1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      manual_acquire_tile(events, tile_id, tiles, n_pes);
    }
    __syncthreads();
    manual_reduce_tile(partial, out, tile_id, m, n, tile_m, tile_n, my_pe,
                       n_pes);
  }
}

std::int64_t tile_count(gemm_ar_problem problem) {
  return ceil_div(problem.m, problem.tile_m) *
         ceil_div(problem.n, problem.tile_n);
}

} // namespace

extern "C" megacu::status
manual_megakernel_cuda_nvshmem_gemm_allreduce_phased_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem) {
  auto validation =
      validate_problem(driver, a, b, partial, out, events, problem);
  if (validation.code != megacu::status_code::ok ||
      !megacu::cuda::has_stream(driver.launch)) {
    return validation;
  }

  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  auto tiles = tile_count(problem);
  auto n_pes = megacu::nvshmem::team_size(driver.team);
  auto *barriers = static_cast<int *>(events);

  status = cuda_status(
      cudaMemsetAsync(barriers, 0, tiles * n_pes * sizeof(int), stream), 11);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
  if (megacu::nvshmem::has_remote_pes(driver.team)) {
    status = cuda_status(cudaStreamSynchronize(stream), 12);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    nvshmem_barrier_all();
  }
#else
  if (megacu::nvshmem::has_remote_pes(driver.team)) {
    return {megacu::status_code::unsupported, 13,
            "device-side NVSHMEM path was not built"};
  }
#endif

  manual_gemm_allreduce_megakernel<<<2, kThreads, 0, stream>>>(
      a, b, partial, out, barriers, problem.m, problem.n, problem.k,
      problem.tile_m, problem.tile_n, tiles, driver.team.team_my_pe, n_pes);
  status = cuda_status(cudaGetLastError(), 14);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  return {};
}
