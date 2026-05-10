#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm_runtime_recipe.cuh"

#include <cuda_runtime.h>

#include <cstdint>

#include <megacu/backends/nvshmem/cuda_event_tensor.cuh>
#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/platform/cuda/megakernel.cuh>
#include <megacu/runtime/device_persistent.cuh>
#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/execution/seeded_orch.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/scheduler/explicit_asap_device.cuh>

#ifdef MEGACU_AG_GEMM_HAS_DEVICE_NVSHMEM
#include <nvshmem.h>

extern "C" void nvshmem_barrier_all();
#endif

namespace {

constexpr int kThreads = 128;
constexpr int kBlocks = 2;

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

__device__ std::int64_t device_ceil_div(std::int64_t value,
                                        std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::int64_t runtime_event_count(megacu::runtime::task_arena_view arena) {
  if (arena.event_count != 0) {
    return static_cast<std::int64_t>(arena.event_count);
  }
  if (arena.event_capacity != 0) {
    return static_cast<std::int64_t>(arena.event_capacity);
  }
  return 1;
}

__device__ std::int64_t k_slice_begin(std::int64_t global_k, int n_pes,
                                      int pe) {
  auto base = global_k / n_pes;
  auto extra = global_k % n_pes;
  return static_cast<std::int64_t>(pe) * base +
         (static_cast<std::int64_t>(pe) < extra ? pe : extra);
}

__device__ std::int64_t k_slice_count(std::int64_t global_k, int n_pes,
                                      int pe) {
  auto base = global_k / n_pes;
  auto extra = global_k % n_pes;
  return base + (static_cast<std::int64_t>(pe) < extra ? 1 : 0);
}

__device__ int owner_for_k(std::int64_t global_k, int n_pes, std::int64_t kk) {
  for (int pe = 0; pe < n_pes; ++pe) {
    auto begin = k_slice_begin(global_k, n_pes, pe);
    auto count = k_slice_count(global_k, n_pes, pe);
    if (kk >= begin && kk < begin + count) {
      return pe;
    }
  }
  return n_pes - 1;
}

struct allgather_tile_produce_task {
  ag_gemm::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(args.problem.n, args.problem.tile_n);
    auto tile_k = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto k_begin = tile_k * args.problem.tile_k;
    auto col_begin = tile_col * args.problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(args.problem.tile_k) * args.problem.tile_n;

    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_k = offset / args.problem.tile_n;
      auto local_col = offset % args.problem.tile_n;
      auto kk = k_begin + local_k;
      auto col = col_begin + local_col;
      if (kk >= args.problem.k || col >= args.problem.n) {
        continue;
      }

      auto n_pes = args.driver.team.team_n_pes;
      auto my_pe = args.driver.team.team_my_pe;
      auto owner = owner_for_k(args.problem.k, n_pes, kk);
      auto owner_begin = k_slice_begin(args.problem.k, n_pes, owner);
      auto local_index = (kk - owner_begin) * args.problem.n + col;
      float value = 0.0F;
      if (owner == my_pe) {
        value = args.b[local_index];
      } else {
#ifdef MEGACU_AG_GEMM_HAS_DEVICE_NVSHMEM
        value = megacu::nvshmem::device::remote_float(args.b, local_index,
                                                      owner);
#endif
      }
      args.gathered_b[kk * args.problem.n + col] =
          value;
    }
  }
};

struct gemm_tile_consume_task {
  ag_gemm::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(args.problem.n, args.problem.tile_n);
    auto tile_row = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto row_begin = tile_row * args.problem.tile_m;
    auto col_begin = tile_col * args.problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(args.problem.tile_m) * args.problem.tile_n;

    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_row = offset / args.problem.tile_n;
      auto local_col = offset % args.problem.tile_n;
      auto row = row_begin + local_row;
      auto col = col_begin + local_col;
      if (row >= args.problem.m || col >= args.problem.n) {
        continue;
      }

      float value = 0.0F;
      for (std::int64_t kk = 0; kk < args.problem.k; ++kk) {
        value += args.a[row * args.problem.k + kk] *
                 args.gathered_b[kk * args.problem.n + col];
      }
      args.out[row * args.problem.n + col] = value;
    }
  }
};

struct allgather_gemm_operators {
  allgather_tile_produce_task allgather;
  gemm_tile_consume_task gemm;

  template <class Context, class Arena, class Task, class Work>
  __device__ void invoke(Context ctx, Arena arena, Task task, Work work) const {
    auto slot = arena.tasks[task.value].op_slot;
    if (slot.value == ag_gemm::runtime_slots::allgather_tile_produce.value) {
      allgather(ctx, work.tile_id);
    } else if (slot.value == ag_gemm::runtime_slots::gemm_tile_consume.value) {
      gemm(ctx, work.tile_id);
    }
  }
};

struct seeded_ag_gemm_recipe {
  ag_gemm::runtime_args args;

  template <class Orch> __device__ megacu::status operator()(Orch &orch) const {
    auto status = ag_gemm::build_runtime_recipe(orch, args);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    return orch.seal();
  }
};

template <class ExecutionModel>
using ag_gemm_runtime = megacu::runtime::device_persistent<
    ExecutionModel, megacu::runtime::loop::block_tile,
    megacu::scheduler::device::explicit_asap,
    megacu::dispatcher::device::tile_grid,
    megacu::backend::nvshmem::cuda::attr_event_tensor_i32,
    allgather_gemm_operators>;

template <class ExecutionModel>
megacu::status launch_runtime(ag_gemm_driver driver,
                              ag_gemm::runtime_args args,
                              megacu::runtime::task_arena_view arena,
                              ExecutionModel execution) {
  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  auto tiles = ag_gemm::tile_k_cols(args.problem) * ag_gemm::tile_cols(args.problem);
  auto n_pes = megacu::nvshmem::team_size(driver.team);
  auto event_count = runtime_event_count(arena);
  auto *barriers = static_cast<int *>(args.events);

  status = cuda_status(cudaMemsetAsync(
                           barriers, 0,
                           event_count * tiles * n_pes * sizeof(int), stream),
                       11);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

#ifdef MEGACU_AG_GEMM_HAS_DEVICE_NVSHMEM
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

  auto runtime = ag_gemm_runtime<ExecutionModel>{
      .execution = execution,
      .loop = {},
      .scheduler = {},
      .dispatcher = {},
      .event_tensor = {.event_tensor = {.events = barriers,
                                        .event_count = event_count,
                                        .tiles = tiles,
                                        .my_pe = driver.team.team_my_pe,
                                        .n_pes = n_pes}},
      .operators = {.allgather = {.args = args}, .gemm = {.args = args}}};

  return cuda_status(megacu::platform::cuda::launch_megakernel(
                         stream, {.blocks = kBlocks, .threads = kThreads},
                         runtime),
                     14);
}

} // namespace

extern "C" megacu::status megacu_cuda_allgather_gemm_host_orch_f32(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, void *events, ag_gemm_problem problem,
    megacu::runtime::task_arena_view arena) {
  if (!megacu::cuda::has_stream(driver.launch)) {
    return {megacu::status_code::invalid_argument, 1,
            "CUDA stream is required"};
  }
  auto args = ag_gemm::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .gathered_b = gathered_b,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  return launch_runtime(
      driver, args, arena,
      megacu::runtime::execution::host_orch::model{.arena = arena});
}

extern "C" megacu::status megacu_cuda_allgather_gemm_seeded_orch_f32(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, void *events, ag_gemm_problem problem,
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    std::uint32_t *construction_status) {
  if (!megacu::cuda::has_stream(driver.launch)) {
    return {megacu::status_code::invalid_argument, 1,
            "CUDA stream is required"};
  }
  auto args = ag_gemm::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .gathered_b = gathered_b,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  auto execution =
      megacu::runtime::execution::seeded_orch::model<seeded_ag_gemm_recipe>{
          .arena = arena,
          .recipe = seeded_ag_gemm_recipe{.args = args},
          .device_status = device_status,
          .construction_status = construction_status};
  return launch_runtime(driver, args, arena, execution);
}
