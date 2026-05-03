#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

#include <cuda_runtime.h>

#include <cstdint>

#include <megacu/backends/nvshmem/cuda_event_tensor.cuh>
#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/platform/cuda/megakernel.cuh>
#include <megacu/runtime/device_entry.cuh>
#include <megacu/scheduler/explicit_asap_device.cuh>

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

struct gemm_tile_produce_task {
  float const *a = nullptr;
  float const *b = nullptr;
  float volatile *partial = nullptr;
  gemm_ar_problem problem;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(problem.n, problem.tile_n);
    auto tile_row = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto row_begin = tile_row * problem.tile_m;
    auto col_begin = tile_col * problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(problem.tile_m) * problem.tile_n;

    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_row = offset / problem.tile_n;
      auto local_col = offset % problem.tile_n;
      auto row = row_begin + local_row;
      auto col = col_begin + local_col;
      if (row >= problem.m || col >= problem.n) {
        continue;
      }

      float value = 0.0f;
      for (std::int64_t kk = 0; kk < problem.k; ++kk) {
        value += a[row * problem.k + kk] * b[kk * problem.n + col];
      }
      partial[row * problem.n + col] = value;
    }
  }
};

struct allreduce_tile_consume_task {
  float volatile const *partial = nullptr;
  float *out = nullptr;
  gemm_ar_problem problem;
  int my_pe = 0;
  int n_pes = 1;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(problem.n, problem.tile_n);
    auto tile_row = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto row_begin = tile_row * problem.tile_m;
    auto col_begin = tile_col * problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(problem.tile_m) * problem.tile_n;

    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_row = offset / problem.tile_n;
      auto local_col = offset % problem.tile_n;
      auto row = row_begin + local_row;
      auto col = col_begin + local_col;
      if (row >= problem.m || col >= problem.n) {
        continue;
      }

      auto index = row * problem.n + col;
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
};

template <class ProducerTask, class ConsumerTask>
struct gemm_allreduce_operators {
  ProducerTask producer;
  ConsumerTask consumer;

  template <class Context, class Task, class Work>
  __device__ void invoke(Context ctx, Task task, Work work) const {
    if (task.value == 0) {
      producer(ctx, work.tile_id);
    } else if (task.value == 2) {
      consumer(ctx, work.tile_id);
    }
  }
};

std::int64_t tile_count(gemm_ar_problem problem) {
  return ceil_div(problem.m, problem.tile_m) *
         ceil_div(problem.n, problem.tile_n);
}

} // namespace

extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
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
  auto *barriers = static_cast<int *>(events);

  status = cuda_status(
      cudaMemsetAsync(barriers, 0,
                      tiles * megacu::nvshmem::team_size(driver.team) *
                          sizeof(int),
                      stream),
      11);
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

  auto n_pes = megacu::nvshmem::team_size(driver.team);
  auto producer = gemm_tile_produce_task{
      .a = a, .b = b, .partial = partial, .problem = problem};
  auto event_tensor = megacu::backend::nvshmem::cuda::task_event_tensor_i32{
      .event_tensor = {.events = barriers,
                       .tiles = tiles,
                       .my_pe = driver.team.team_my_pe,
                       .n_pes = n_pes},
      .publish_task = 0,
      .acquire_task = 2};
  auto dispatcher = megacu::dispatcher::device::tile_grid{.tiles = tiles};
  auto scheduler = megacu::scheduler::device::explicit_asap{.task_count = 3};
  auto consumer = allreduce_tile_consume_task{.partial = partial,
                                              .out = out,
                                              .problem = problem,
                                              .my_pe = driver.team.team_my_pe,
                                              .n_pes = n_pes};
  auto operators =
      gemm_allreduce_operators{.producer = producer, .consumer = consumer};
  auto entry = megacu::runtime::device::entry{.scheduler = scheduler,
                                              .dispatcher = dispatcher,
                                              .backend = event_tensor,
                                              .operators = operators};

  status = cuda_status(megacu::platform::cuda::launch_megakernel(
                           stream, {.blocks = 2, .threads = kThreads}, entry),
                       14);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  return {};
}
