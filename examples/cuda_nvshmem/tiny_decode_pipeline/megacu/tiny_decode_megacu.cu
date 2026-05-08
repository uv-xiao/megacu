#include "examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_arena.cuh"

#include <cuda_runtime.h>

#include <cstdint>

#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/platform/cuda.h>
#include <megacu/platform/cuda/local_event_tensor.cuh>
#include <megacu/platform/cuda/megakernel.cuh>
#include <megacu/runtime/device_persistent.cuh>
#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/execution/seeded_orch.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/scheduler/explicit_asap_device.cuh>

namespace {

namespace tiny = megacu::examples::tiny_decode;
namespace decode_runtime = tiny_decode_runtime;

constexpr int kThreads = 32;
constexpr int kBlocks = 4;
constexpr std::uint32_t kTaskCapacity = decode_runtime::task_capacity;
constexpr std::uint32_t kEventCapacity = decode_runtime::event_capacity;
constexpr std::uint32_t kDepCapacity = decode_runtime::dep_capacity;
constexpr std::uint32_t kRegionCapacity = decode_runtime::region_capacity;
constexpr std::int64_t kMaxEventTiles = decode_runtime::max_event_tiles;

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

template <class T>
megacu::status allocate_managed(T *&ptr, std::size_t count,
                                std::uint16_t detail) {
  void *raw = nullptr;
  auto status = cuda_status(cudaMallocManaged(&raw, sizeof(T) * count), detail);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  ptr = static_cast<T *>(raw);
  return {};
}

struct managed_arena {
  megacu::runtime::task_arena_view view;

  megacu::status allocate() {
    view.task_capacity = kTaskCapacity;
    view.event_capacity = kEventCapacity;
    view.dep_capacity = kDepCapacity;
    view.region_capacity = kRegionCapacity;

    auto status = allocate_managed(view.tasks, view.task_capacity, 20);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.events, view.event_capacity, 21);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.deps, view.dep_capacity, 22);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.regions, view.region_capacity, 23);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.task_completed, view.task_capacity, 24);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    return allocate_managed(view.task_remaining_work, view.task_capacity, 25);
  }

  void release() {
    cudaFree(view.task_remaining_work);
    cudaFree(view.task_completed);
    cudaFree(view.regions);
    cudaFree(view.deps);
    cudaFree(view.events);
    cudaFree(view.tasks);
    view = {};
  }
};

void clear_arena(managed_arena &arena) {
  arena.view.task_count = 0;
  arena.view.event_count = 0;
  arena.view.dep_count = 0;
  arena.view.region_count = 0;
  for (std::uint32_t index = 0; index < arena.view.task_capacity; ++index) {
    arena.view.tasks[index] = {};
    arena.view.task_completed[index] = 0;
    arena.view.task_remaining_work[index] = 0;
  }
  for (std::uint32_t index = 0; index < arena.view.event_capacity; ++index) {
    arena.view.events[index] = {};
  }
  for (std::uint32_t index = 0; index < arena.view.dep_capacity; ++index) {
    arena.view.deps[index] = {};
  }
  for (std::uint32_t index = 0; index < arena.view.region_capacity; ++index) {
    arena.view.regions[index] = {};
  }
}

void copy_host_arena(decode_runtime::host_arena_storage const &host,
                     managed_arena &device) {
  device.view.task_count = host.task_count;
  device.view.event_count = host.event_count;
  device.view.dep_count = host.dep_count;
  device.view.region_count = host.region_count;
  for (std::uint32_t index = 0; index < device.view.task_capacity; ++index) {
    device.view.tasks[index] = host.tasks[index];
    device.view.task_completed[index] = host.completed[index];
    device.view.task_remaining_work[index] = host.remaining[index];
  }
  for (std::uint32_t index = 0; index < device.view.event_capacity; ++index) {
    device.view.events[index] = host.events[index];
  }
  for (std::uint32_t index = 0; index < device.view.dep_capacity; ++index) {
    device.view.deps[index] = host.deps[index];
  }
  for (std::uint32_t index = 0; index < device.view.region_capacity; ++index) {
    device.view.regions[index] = host.regions[index];
  }
}

struct norm_task {
  decode_runtime::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    if (ctx.thread_id() == 0 &&
        tile_id < static_cast<std::int64_t>(tiny::hidden_size)) {
      args.state->norm[tile_id] = args.state->hidden[tile_id] * 0.5F;
    }
  }
};

struct projection_task {
  decode_runtime::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    if (ctx.thread_id() == 0 &&
        tile_id < static_cast<std::int64_t>(tiny::hidden_size)) {
      args.state->projection[tile_id] =
          args.state->norm[tile_id] + static_cast<float>(tile_id + 1);
    }
  }
};

struct residual_task {
  decode_runtime::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    if (ctx.thread_id() == 0 &&
        tile_id < static_cast<std::int64_t>(tiny::hidden_size)) {
      args.state->residual[tile_id] =
          args.state->projection[tile_id] + args.state->hidden[tile_id];
    }
  }
};

struct mlp_task {
  decode_runtime::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    if (ctx.thread_id() == 0 &&
        tile_id < static_cast<std::int64_t>(tiny::mlp_size)) {
      auto value = args.state->residual[tile_id];
      args.state->mlp[tile_id] = value * value * 0.25F;
    }
  }
};

struct logits_task {
  decode_runtime::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    if (ctx.thread_id() != 0 ||
        tile_id >= static_cast<std::int64_t>(tiny::vocab_size)) {
      return;
    }

    float sum = 0.0F;
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(tiny::mlp_size); ++index) {
      sum += args.state->mlp[index] *
             (static_cast<float>((index + 1) * (tile_id + 1)) * 0.125F);
    }
    args.state->logits[tile_id] = sum;
  }
};

struct tiny_decode_operators {
  norm_task norm;
  projection_task projection;
  residual_task residual;
  mlp_task mlp;
  logits_task logits;

  template <class Context, class Arena, class Task, class Work>
  __device__ void invoke(Context ctx, Arena arena, Task task, Work work) const {
    auto slot = arena.tasks[task.value].op_slot;
    if (slot.value == decode_runtime::slots::norm.value) {
      norm(ctx, work.tile_id);
    } else if (slot.value == decode_runtime::slots::projection.value) {
      projection(ctx, work.tile_id);
    } else if (slot.value == decode_runtime::slots::residual.value) {
      residual(ctx, work.tile_id);
    } else if (slot.value == decode_runtime::slots::mlp.value) {
      mlp(ctx, work.tile_id);
    } else if (slot.value == decode_runtime::slots::logits.value) {
      logits(ctx, work.tile_id);
    }
  }
};

struct seeded_decode_recipe {
  decode_runtime::runtime_args args;

  template <class Orch> __device__ megacu::status operator()(Orch &orch) const {
    auto status = decode_runtime::build_recipe(orch, args);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    return orch.seal();
  }
};

template <class ExecutionModel>
using tiny_decode_runtime = megacu::runtime::device_persistent<
    ExecutionModel, megacu::runtime::loop::block_tile,
    megacu::scheduler::device::explicit_asap,
    megacu::dispatcher::device::tile_grid,
    megacu::platform::cuda::local_event_tensor_i32,
    tiny_decode_operators>;

template <class ExecutionModel>
megacu::status launch_runtime(decode_runtime::runtime_args args,
                              megacu::runtime::task_arena_view arena,
                              ExecutionModel execution, int *events) {
  auto status = cuda_status(cudaMemset(events, 0,
                                       kEventCapacity * kMaxEventTiles *
                                           sizeof(int)),
                            30);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto runtime = tiny_decode_runtime<ExecutionModel>{
      .execution = execution,
      .loop = {},
      .scheduler = {},
      .dispatcher = {},
      .event_tensor = {.events = events, .max_tiles = kMaxEventTiles},
      .operators = {.norm = {.args = args},
                    .projection = {.args = args},
                    .residual = {.args = args},
                    .mlp = {.args = args},
                    .logits = {.args = args}}};

  return cuda_status(megacu::platform::cuda::launch_megakernel(
                         nullptr, {.blocks = kBlocks, .threads = kThreads},
                         runtime),
                     31);
}

megacu::status run_with_managed_state(
    tiny::buffers *state,
    megacu::status (*launch)(decode_runtime::runtime_args, managed_arena &,
                             int *)) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  tiny::buffers *device_state = nullptr;
  auto status = allocate_managed(device_state, 1, 40);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  *device_state = *state;

  int *events = nullptr;
  status = allocate_managed(events, kEventCapacity * kMaxEventTiles, 41);
  if (status.code != megacu::status_code::ok) {
    cudaFree(device_state);
    return status;
  }

  managed_arena arena;
  status = arena.allocate();
  if (status.code != megacu::status_code::ok) {
    arena.release();
    cudaFree(events);
    cudaFree(device_state);
    return status;
  }

  status = launch({.state = device_state}, arena, events);
  if (status.code == megacu::status_code::ok) {
    status = cuda_status(cudaDeviceSynchronize(), 42);
  }
  if (status.code == megacu::status_code::ok) {
    *state = *device_state;
  }

  arena.release();
  cudaFree(events);
  cudaFree(device_state);
  return status;
}

megacu::status launch_host_orch(decode_runtime::runtime_args args,
                                managed_arena &arena, int *events) {
  decode_runtime::host_arena_storage host_arena;
  auto status = decode_runtime::build_host_arena(host_arena, args);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  copy_host_arena(host_arena, arena);
  return launch_runtime(args, arena.view,
                        megacu::runtime::execution::host_orch::model{
                            .arena = arena.view},
                        events);
}

megacu::status launch_seeded_orch(decode_runtime::runtime_args args,
                                  managed_arena &arena, int *events) {
  clear_arena(arena);

  megacu::status *device_status = nullptr;
  std::uint32_t *construction_status = nullptr;
  auto status = allocate_managed(device_status, 1, 50);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  status = allocate_managed(construction_status, 1, 51);
  if (status.code != megacu::status_code::ok) {
    cudaFree(device_status);
    return status;
  }
  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;

  auto execution =
      megacu::runtime::execution::seeded_orch::model<seeded_decode_recipe>{
          .arena = arena.view,
          .recipe = seeded_decode_recipe{.args = args},
          .device_status = device_status,
          .construction_status = construction_status};
  status = launch_runtime(args, arena.view, execution, events);
  if (status.code == megacu::status_code::ok) {
    status = cuda_status(cudaDeviceSynchronize(), 52);
  }
  if (status.code == megacu::status_code::ok &&
      device_status->code != megacu::status_code::ok) {
    status = *device_status;
  }

  cudaFree(construction_status);
  cudaFree(device_status);
  return status;
}

int status_to_result(megacu::status status) {
  return status.code == megacu::status_code::ok ? 0 : 1;
}

} // namespace

extern "C" int megacu_tiny_decode_megacu_host_orch(tiny::buffers *state) {
  return status_to_result(run_with_managed_state(state, &launch_host_orch));
}

extern "C" int megacu_tiny_decode_megacu_seeded_orch(tiny::buffers *state) {
  return status_to_result(run_with_managed_state(state, &launch_seeded_orch));
}

extern "C" int megacu_tiny_decode_megacu(tiny::buffers *state) {
  return megacu_tiny_decode_megacu_host_orch(state);
}
