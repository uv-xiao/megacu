#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include <megacu/backends/nvshmem/cuda_event_tensor.cuh>
#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/task_arena.h>
#include <megacu/scheduler/explicit_asap_device.cuh>

namespace {

struct arena_runtime_context {
  __device__ int block_id() const { return blockIdx.x; }
  __device__ int grid_blocks() const { return gridDim.x; }
  __device__ int thread_id() const { return threadIdx.x; }
  __device__ void sync_block() const { __syncthreads(); }
};

struct arena_runtime_observation {
  int first_task = -1;
  int first_work_active = 0;
  int first_work_tile = -1;
  int next_work_active = 0;
  int next_work_tile = -1;
  int next_ready_before_completion = -2;
  int task0_remaining_after_one = -1;
  int task0_completed = 0;
  int ready_after_one_completion = -2;
  int second_task = -1;
  int second_work_active = 0;
  int second_work_tile = -1;
};

struct op_observation {
  int producer_count = 0;
  int consumer_count = 0;
  int unexpected_operator_count = 0;
  int consumer_after_sync = 0;
};

struct event_wait_observation {
  int wait_entered = 0;
  int release_seen = 0;
  int complete_returned = 0;
  int complete_result = 0;
};

struct contract_operators {
  op_observation *obs = nullptr;

  template <class Context, class Arena, class Task, class Work>
  __device__ void invoke(Context ctx, Arena arena, Task task, Work) const {
    if (ctx.thread_id() != 0) {
      return;
    }
    auto const slot = arena.tasks[task.value].op_slot;
    if (slot == megacu::runtime::operator_slot{3}) {
      atomicAdd(&obs->producer_count, 1);
    } else if (slot == megacu::runtime::operator_slot{4}) {
      if (arena.task_completed[1] == 1) {
        atomicAdd(&obs->consumer_after_sync, 1);
      }
      atomicAdd(&obs->consumer_count, 1);
    } else {
      atomicAdd(&obs->unexpected_operator_count, 1);
    }
  }
};

__global__ void
observe_first_arena_work(megacu::runtime::task_arena_view arena,
                         arena_runtime_observation *observations) {
  auto ctx = arena_runtime_context{};
  auto scheduler = megacu::scheduler::device::explicit_asap{};
  auto dispatcher = megacu::dispatcher::device::tile_grid{};
  auto work = megacu::runtime::device::work_item{};

  auto task = scheduler.first(ctx, arena, work);
  auto first_work = dispatcher.first(ctx, arena, task);
  auto next_work = dispatcher.next(ctx, arena, task, first_work);

  if (ctx.thread_id() == 0) {
    auto &observation = observations[ctx.block_id()];
    observation.first_task = task.value;
    observation.first_work_active = first_work.active ? 1 : 0;
    observation.first_work_tile = static_cast<int>(first_work.tile_id);
    observation.next_work_active = next_work.active ? 1 : 0;
    observation.next_work_tile = static_cast<int>(next_work.tile_id);
    observation.next_ready_before_completion =
        scheduler.next(ctx, arena, work, task).value;
  }
}

__global__ void
complete_one_producer_work(megacu::runtime::task_arena_view arena,
                           arena_runtime_observation *observation) {
  auto ctx = arena_runtime_context{};
  auto scheduler = megacu::scheduler::device::explicit_asap{};
  auto dispatcher = megacu::dispatcher::device::tile_grid{};
  auto work = megacu::runtime::device::work_item{};
  auto task = scheduler.first(ctx, arena, work);
  auto first_work = dispatcher.first(ctx, arena, task);

  if (ctx.thread_id() == 0) {
    scheduler.complete(ctx, arena, task, first_work);
    observation->task0_remaining_after_one =
        static_cast<int>(arena.task_remaining_work[0]);
    observation->task0_completed = static_cast<int>(arena.task_completed[0]);
    observation->ready_after_one_completion =
        scheduler.next(ctx, arena, work, task).value;
  }
}

__global__ void
complete_all_producer_work(megacu::runtime::task_arena_view arena) {
  auto ctx = arena_runtime_context{};
  auto scheduler = megacu::scheduler::device::explicit_asap{};
  auto dispatcher = megacu::dispatcher::device::tile_grid{};
  auto work = megacu::runtime::device::work_item{};
  auto task = scheduler.first(ctx, arena, work);

  for (auto cursor = dispatcher.first(ctx, arena, task); cursor.active;
       cursor = dispatcher.next(ctx, arena, task, cursor)) {
    if (ctx.thread_id() == 0) {
      scheduler.complete(ctx, arena, task, cursor);
    }
  }
}

__global__ void
observe_ready_after_producer(megacu::runtime::task_arena_view arena,
                             arena_runtime_observation *observation) {
  auto ctx = arena_runtime_context{};
  auto scheduler = megacu::scheduler::device::explicit_asap{};
  auto dispatcher = megacu::dispatcher::device::tile_grid{};
  auto work = megacu::runtime::device::work_item{};
  auto current = megacu::runtime::device::task_ref{.value = 0};
  auto second_task = scheduler.next(ctx, arena, work, current);
  auto second_work = dispatcher.first(ctx, arena, second_task);

  if (ctx.thread_id() == 0) {
    observation->task0_remaining_after_one =
        static_cast<int>(arena.task_remaining_work[0]);
    observation->task0_completed = static_cast<int>(arena.task_completed[0]);
    observation->second_task = second_task.value;
    observation->second_work_active = second_work.active ? 1 : 0;
    observation->second_work_tile = static_cast<int>(second_work.tile_id);
  }
}

__global__ void reset_task0_completion(megacu::runtime::task_arena_view arena) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    arena.task_completed[0] = 0;
    arena.task_remaining_work[0] = 4;
  }
}

__global__ void
execute_arena_runtime(megacu::runtime::task_arena_view arena,
                      int *event_storage,
                      op_observation *observation) {
  auto ctx = arena_runtime_context{};
  megacu::runtime::loop::block_tile{}.run(
      ctx, arena, megacu::scheduler::device::explicit_asap{},
      megacu::dispatcher::device::tile_grid{},
      megacu::backend::nvshmem::cuda::attr_event_tensor_i32{
          .event_tensor = {.events = event_storage,
                           .tiles = 1,
                           .my_pe = 0,
                           .n_pes = 1}},
      contract_operators{observation});
}

__global__ void
observe_attr_event_wait(megacu::runtime::task_arena_view arena,
                        int *event_storage,
                        event_wait_observation *observation) {
  auto ctx = arena_runtime_context{};
  auto event_tensor =
      megacu::backend::nvshmem::cuda::attr_event_tensor_i32{
          .event_tensor = {.events = event_storage,
                           .tiles = 1,
                           .my_pe = 0,
                           .n_pes = 1}};
  auto work = megacu::runtime::device::work_item{.active = true, .tile_id = 0};
  auto sync_task = megacu::runtime::device::task_ref{.value = 1};

  if (ctx.thread_id() == 0) {
    megacu::cuda::device::signal_ready(&observation->wait_entered, 0, 1);
    auto const complete = event_tensor.complete(ctx, arena, sync_task, work);
    observation->complete_result = complete ? 1 : 0;
    observation->release_seen =
        megacu::cuda::device::load_ready(&observation->release_seen, 0);
    observation->complete_returned = 1;
    assert(complete);
    assert(observation->release_seen == 1);
    assert(event_storage[0] == 1);
  } else if (ctx.thread_id() == 32) {
    megacu::cuda::device::wait_ready(&observation->wait_entered, 0, 1);
    for (int spin = 0; spin < 1024; ++spin) {
      if (megacu::cuda::device::load_ready(&observation->complete_returned,
                                           0) != 0) {
        break;
      }
    }
    assert(megacu::cuda::device::load_ready(&observation->complete_returned,
                                            0) == 0);
    megacu::cuda::device::signal_ready(&observation->release_seen, 0, 1);
    megacu::backend::nvshmem::cuda::event_tensor_i32{
        .events = event_storage, .tiles = 1, .my_pe = 0, .n_pes = 1}
        .notify(0, 1);
  }
}

} // namespace

int main() {
  megacu::runtime::device_task_record *tasks = nullptr;
  megacu::runtime::device_event_tensor_record *events = nullptr;
  megacu::runtime::task_ref *deps = nullptr;
  megacu::runtime::arena_region *regions = nullptr;
  std::uint32_t *completed = nullptr;
  std::uint32_t *remaining = nullptr;
  int *event_storage = nullptr;
  arena_runtime_observation *observations = nullptr;
  op_observation *op_observations = nullptr;
  event_wait_observation *event_wait_observations = nullptr;

  assert(cudaMallocManaged(&tasks, sizeof(*tasks) * 3) == cudaSuccess);
  assert(cudaMallocManaged(&events, sizeof(*events) * 1) == cudaSuccess);
  assert(cudaMallocManaged(&deps, sizeof(*deps) * 2) == cudaSuccess);
  assert(cudaMallocManaged(&regions, sizeof(*regions) * 1) == cudaSuccess);
  assert(cudaMallocManaged(&completed, sizeof(*completed) * 3) == cudaSuccess);
  assert(cudaMallocManaged(&remaining, sizeof(*remaining) * 3) == cudaSuccess);
  assert(cudaMallocManaged(&event_storage, sizeof(*event_storage)) ==
         cudaSuccess);
  assert(cudaMallocManaged(&observations, sizeof(*observations) * 2) ==
         cudaSuccess);
  assert(cudaMallocManaged(&op_observations, sizeof(*op_observations)) ==
         cudaSuccess);
  assert(cudaMallocManaged(&event_wait_observations,
                           sizeof(*event_wait_observations)) == cudaSuccess);

  events[0] = {.ref = megacu::runtime::event_tensor_ref{0},
               .attributes = megacu::runtime::attrs(
                   megacu::runtime::event_tensor::shape(1, 1),
                   megacu::runtime::event_tensor::wait_count(1))};
  tasks[0] = {.kind = megacu::runtime::task_kind::operator_body,
              .op_slot = megacu::runtime::operator_slot{3},
              .dep_count = 0,
              .first_dep = 0,
              .attributes = megacu::runtime::attrs(
                  megacu::runtime::dispatcher::tile_grid(2, 2),
                  megacu::runtime::event_tensor::notify(
                      megacu::runtime::event_tensor_ref{0}))};
  tasks[1] = {
      .kind = megacu::runtime::task_kind::sync_only,
      .op_slot = megacu::runtime::invalid_operator_slot(),
      .dep_count = 1,
      .first_dep = 0,
      .attributes = megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(megacu::runtime::task_ref{0}),
          megacu::runtime::event_tensor::wait(
              megacu::runtime::event_tensor_ref{0}))};
  tasks[2] = {
      .kind = megacu::runtime::task_kind::operator_body,
      .op_slot = megacu::runtime::operator_slot{4},
      .dep_count = 1,
      .first_dep = 1,
      .attributes = megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(megacu::runtime::task_ref{1}),
          megacu::runtime::dispatcher::tile_grid(1, 1))};
  deps[0] = megacu::runtime::task_ref{0};
  deps[1] = megacu::runtime::task_ref{1};
  regions[0] = {.epoch = 0,
                .first_task = 0,
                .task_count = 3,
                .first_event = 0,
                .event_count = 1,
                .sealed = 1};
  completed[0] = 0;
  completed[1] = 0;
  completed[2] = 0;
  remaining[0] = 4;
  remaining[1] = 1;
  remaining[2] = 1;
  event_storage[0] = 0;
  observations[0] = {};
  observations[1] = {};

  auto arena =
      megacu::runtime::task_arena_view{.tasks = tasks,
                                       .events = events,
                                       .deps = deps,
                                       .regions = regions,
                                       .task_completed = completed,
                                       .task_remaining_work = remaining,
                                       .task_count = 3,
                                       .event_count = 1,
                                       .dep_count = 2,
                                       .region_count = 1,
                                       .task_capacity = 3,
                                       .event_capacity = 1,
                                       .dep_capacity = 2,
                                       .region_capacity = 1};

  observe_first_arena_work<<<2, 1>>>(arena, observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(observations[0].first_task == 0);
  assert(observations[0].first_work_active == 1);
  assert(observations[0].first_work_tile == 0);
  assert(observations[0].next_work_active == 1);
  assert(observations[0].next_work_tile == 2);
  assert(observations[0].next_ready_before_completion == -1);
  assert(observations[1].first_task == 0);
  assert(observations[1].first_work_active == 1);
  assert(observations[1].first_work_tile == 1);
  assert(observations[1].next_work_active == 1);
  assert(observations[1].next_work_tile == 3);
  assert(observations[1].next_ready_before_completion == -1);

  observations[0] = {};
  complete_one_producer_work<<<1, 1>>>(arena, observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(observations[0].task0_remaining_after_one == 3);
  assert(observations[0].task0_completed == 0);
  assert(observations[0].ready_after_one_completion == -1);

  reset_task0_completion<<<1, 1>>>(arena);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  observations[0] = {};
  complete_all_producer_work<<<2, 1>>>(arena);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  observe_ready_after_producer<<<1, 1>>>(arena, observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(observations[0].task0_remaining_after_one == 0);
  assert(observations[0].task0_completed == 1);
  assert(observations[0].second_task == 1);
  assert(observations[0].second_work_active == 1);
  assert(observations[0].second_work_tile == 0);

  event_storage[0] = 0;
  *event_wait_observations = {};
  observe_attr_event_wait<<<1, 64>>>(arena, event_storage,
                                     event_wait_observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(event_wait_observations->wait_entered == 1);
  assert(event_wait_observations->release_seen == 1);
  assert(event_wait_observations->complete_returned == 1);
  assert(event_wait_observations->complete_result == 1);
  assert(event_storage[0] == 1);

  completed[0] = 0;
  completed[1] = 0;
  completed[2] = 0;
  remaining[0] = 4;
  remaining[1] = 1;
  remaining[2] = 1;
  event_storage[0] = 0;
  *op_observations = {};

  execute_arena_runtime<<<2, 4>>>(arena, event_storage, op_observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(op_observations->producer_count == 4);
  assert(op_observations->unexpected_operator_count == 0);
  assert(event_storage[0] == 1);
  assert(op_observations->consumer_count == 1);
  assert(op_observations->consumer_after_sync == 1);
  assert(completed[0] == 1);
  assert(completed[1] == 1);
  assert(completed[2] == 1);
  assert(remaining[0] == 0);
  assert(remaining[1] == 0);
  assert(remaining[2] == 0);

  assert(cudaFree(event_wait_observations) == cudaSuccess);
  assert(cudaFree(op_observations) == cudaSuccess);
  assert(cudaFree(observations) == cudaSuccess);
  assert(cudaFree(event_storage) == cudaSuccess);
  assert(cudaFree(remaining) == cudaSuccess);
  assert(cudaFree(completed) == cudaSuccess);
  assert(cudaFree(regions) == cudaSuccess);
  assert(cudaFree(deps) == cudaSuccess);
  assert(cudaFree(events) == cudaSuccess);
  assert(cudaFree(tasks) == cudaSuccess);
  return 0;
}
