#include <cassert>

#include <cuda_runtime.h>

#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/runtime/device_persistent.cuh>
#include <megacu/runtime/execution/seeded_orch.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/task_arena.h>
#include <megacu/scheduler/explicit_asap_device.cuh>

namespace {

struct fake_context {};

struct fake_execution {
  megacu::runtime::task_arena_view arena;

  __device__ megacu::runtime::task_arena_view bind(fake_context) const {
    return arena;
  }

  __device__ void construct(fake_context,
                            megacu::runtime::task_arena_view) const {}
};

struct fake_scheduler {};
struct fake_dispatcher {};
struct fake_event_tensor {};
struct fake_operators {};

struct fake_loop {
  int *marker = nullptr;

  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context, Arena, Scheduler, Dispatcher, EventTensor,
                      Operators) const {
    if (threadIdx.x == 0) {
      *marker = 7;
    }
  }
};

struct seeded_runtime_observation {
  int sealed = 0;
  int task_count = 0;
  int event_count = 0;
  int dep_count = 0;
  int region_count = 0;
};

struct recording_loop {
  seeded_runtime_observation *observations = nullptr;

  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context ctx, Arena arena, Scheduler, Dispatcher,
                      EventTensor, Operators) const {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    observations[index] = {
        .sealed = static_cast<int>(arena.regions[0].sealed),
        .task_count = static_cast<int>(arena.task_count),
        .event_count = static_cast<int>(arena.event_count),
        .dep_count = static_cast<int>(arena.dep_count),
        .region_count = static_cast<int>(arena.region_count)};
  }
};

struct seeded_context {
  __device__ int block_id() const { return blockIdx.x; }
  __device__ int grid_blocks() const { return gridDim.x; }
  __device__ int thread_id() const { return threadIdx.x; }
  __device__ void sync_block() const { __syncthreads(); }
};

struct noop_event_tensor {
  template <class Context, class Task, class Work>
  __device__ void before(Context, Task, Work) const {}

  template <class Context, class Task, class Work>
  __device__ void after(Context, Task, Work) const {}
};

struct marker_operators {
  int *marker = nullptr;

  template <class Context, class Task, class Work>
  __device__ void invoke(Context ctx, Task, Work) const {
    if (ctx.thread_id() == 0) {
      *marker = 99;
    }
  }
};

struct seeded_recipe {
  template <class Orch> __device__ megacu::status operator()(Orch &orch) const {
    auto ready = orch.event_tensor(megacu::runtime::attrs(
        megacu::runtime::event_tensor::shape(1, 1),
        megacu::runtime::event_tensor::wait_count(1)));
    auto producer = orch.submit(
        megacu::runtime::operator_slot{3},
        megacu::runtime::attrs(megacu::runtime::event_tensor::notify(ready)));
    auto sync = orch.sync(megacu::runtime::attrs(
        megacu::runtime::scheduler::depends_on(producer),
        megacu::runtime::event_tensor::wait(ready)));
    orch.submit(megacu::runtime::operator_slot{4},
                megacu::runtime::attrs(
                    megacu::runtime::scheduler::depends_on(sync)));
    return orch.seal();
  }
};

struct task_overflow_seeded_recipe {
  template <class Orch> __device__ megacu::status operator()(Orch &orch) const {
    (void)orch.submit(megacu::runtime::operator_slot{1});
    (void)orch.submit(megacu::runtime::operator_slot{2});
    return orch.current_status();
  }
};

__global__ void construct_seeded_orch(megacu::runtime::task_arena_view arena,
                                      megacu::status *device_status,
                                      std::uint32_t *construction_status) {
  auto execution =
      megacu::runtime::execution::seeded_orch::model<seeded_recipe>{
          .arena = arena, .recipe = seeded_recipe{},
          .device_status = device_status,
          .construction_status = construction_status};
  auto bound = execution.bind(seeded_context{});
  execution.construct(seeded_context{}, bound);
}

__global__ void publish_task_overflow(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    seeded_runtime_observation *observation) {
  megacu::runtime::device_orch orch{arena};
  (void)orch.submit(megacu::runtime::operator_slot{1});
  (void)orch.submit(megacu::runtime::operator_slot{2});
  *device_status = orch.current_status();
  observation[0] = {.task_count = static_cast<int>(arena.task_count),
                    .event_count = static_cast<int>(arena.event_count),
                    .dep_count = static_cast<int>(arena.dep_count),
                    .region_count = static_cast<int>(arena.region_count)};
}

__global__ void publish_dep_overflow(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    seeded_runtime_observation *observation) {
  megacu::runtime::device_orch orch{arena};
  auto first = orch.submit(megacu::runtime::operator_slot{1});
  (void)orch.submit(
      megacu::runtime::operator_slot{2},
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(first),
                             megacu::runtime::scheduler::depends_on(first)));
  *device_status = orch.current_status();
  observation[0] = {.task_count = static_cast<int>(arena.task_count),
                    .event_count = static_cast<int>(arena.event_count),
                    .dep_count = static_cast<int>(arena.dep_count),
                    .region_count = static_cast<int>(arena.region_count)};
}

__global__ void publish_event_overflow(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    seeded_runtime_observation *observation) {
  megacu::runtime::device_orch orch{arena};
  (void)orch.event_tensor();
  (void)orch.event_tensor();
  *device_status = orch.current_status();
  observation[0] = {.task_count = static_cast<int>(arena.task_count),
                    .event_count = static_cast<int>(arena.event_count),
                    .dep_count = static_cast<int>(arena.dep_count),
                    .region_count = static_cast<int>(arena.region_count)};
}

__global__ void publish_region_overflow(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    seeded_runtime_observation *observation) {
  megacu::runtime::device_orch orch{arena};
  (void)orch.event_tensor();
  (void)orch.submit(megacu::runtime::operator_slot{1});
  *device_status = orch.seal();
  observation[0] = {.task_count = static_cast<int>(arena.task_count),
                    .event_count = static_cast<int>(arena.event_count),
                    .dep_count = static_cast<int>(arena.dep_count),
                    .region_count = static_cast<int>(arena.region_count)};
}

__global__ void instantiate_persistent_runtime(int *marker) {
  auto runtime = megacu::runtime::device_persistent<
      fake_execution, fake_loop, fake_scheduler, fake_dispatcher,
      fake_event_tensor, fake_operators>{
      .execution = fake_execution{},
      .loop = fake_loop{.marker = marker},
      .scheduler = fake_scheduler{},
      .dispatcher = fake_dispatcher{},
      .event_tensor = fake_event_tensor{},
      .operators = fake_operators{}};

  runtime(fake_context{});
}

__global__ void instantiate_seeded_persistent_runtime(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    std::uint32_t *construction_status,
    seeded_runtime_observation *observations) {
  auto runtime = megacu::runtime::device_persistent<
      megacu::runtime::execution::seeded_orch::model<seeded_recipe>,
      recording_loop, fake_scheduler, fake_dispatcher, fake_event_tensor,
      fake_operators>{
      .execution = {.arena = arena,
                    .recipe = seeded_recipe{},
                    .device_status = device_status,
                    .construction_status = construction_status},
      .loop = recording_loop{.observations = observations},
      .scheduler = fake_scheduler{},
      .dispatcher = fake_dispatcher{},
      .event_tensor = fake_event_tensor{},
      .operators = fake_operators{}};

  runtime(seeded_context{});
}

__global__ void instantiate_seeded_task_overflow_persistent_runtime(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    std::uint32_t *construction_status,
    seeded_runtime_observation *observations) {
  auto runtime = megacu::runtime::device_persistent<
      megacu::runtime::execution::seeded_orch::model<
          task_overflow_seeded_recipe>,
      recording_loop, fake_scheduler, fake_dispatcher, fake_event_tensor,
      fake_operators>{
      .execution = {.arena = arena,
                    .recipe = task_overflow_seeded_recipe{},
                    .device_status = device_status,
                    .construction_status = construction_status},
      .loop = recording_loop{.observations = observations},
      .scheduler = fake_scheduler{},
      .dispatcher = fake_dispatcher{},
      .event_tensor = fake_event_tensor{},
      .operators = fake_operators{}};

  runtime(seeded_context{});
}

__global__ void instantiate_seeded_failure_block_tile_runtime(
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    std::uint32_t *construction_status, int *marker) {
  auto runtime = megacu::runtime::device_persistent<
      megacu::runtime::execution::seeded_orch::model<
          task_overflow_seeded_recipe>,
      megacu::runtime::loop::block_tile, megacu::scheduler::device::explicit_asap,
      megacu::dispatcher::device::tile_grid, noop_event_tensor,
      marker_operators>{
      .execution = {.arena = arena,
                    .recipe = task_overflow_seeded_recipe{},
                    .device_status = device_status,
                    .construction_status = construction_status},
      .loop = megacu::runtime::loop::block_tile{},
      .scheduler = megacu::scheduler::device::explicit_asap{.task_count = 1},
      .dispatcher = megacu::dispatcher::device::tile_grid{.tiles = 1},
      .event_tensor = noop_event_tensor{},
      .operators = marker_operators{.marker = marker}};

  runtime(seeded_context{});
}

bool has_event_wait(megacu::runtime::device_task_record const &task) {
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_wait) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  static_assert(
      requires {
        typename megacu::runtime::device_persistent<
            fake_execution, fake_loop, fake_scheduler, fake_dispatcher,
            fake_event_tensor, fake_operators>;
      },
      "device_persistent must be the runtime-owned composition point");
  static_assert(requires { typename megacu::runtime::loop::block_tile; },
                "block_tile must live under runtime::loop");
  static_assert(
      requires {
        typename megacu::runtime::execution::seeded_orch::model<seeded_recipe>;
      },
      "seeded_orch must be a runtime execution model");

  int *marker = nullptr;
  auto alloc = cudaMallocManaged(&marker, sizeof(int));
  assert(alloc == cudaSuccess);
  *marker = 0;

  instantiate_persistent_runtime<<<1, 1>>>(marker);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(*marker == 7);

  megacu::runtime::device_task_record *tasks = nullptr;
  megacu::runtime::device_event_tensor_record *events = nullptr;
  megacu::runtime::task_ref *deps = nullptr;
  megacu::runtime::arena_region *regions = nullptr;
  megacu::status *device_status = nullptr;
  std::uint32_t *construction_status = nullptr;
  assert(cudaMallocManaged(&tasks, sizeof(*tasks) * 3) == cudaSuccess);
  assert(cudaMallocManaged(&events, sizeof(*events) * 1) == cudaSuccess);
  assert(cudaMallocManaged(&deps, sizeof(*deps) * 2) == cudaSuccess);
  assert(cudaMallocManaged(&regions, sizeof(*regions) * 1) == cudaSuccess);
  assert(cudaMallocManaged(&device_status, sizeof(*device_status)) ==
         cudaSuccess);
  assert(cudaMallocManaged(&construction_status,
                           sizeof(*construction_status)) == cudaSuccess);

  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;
  regions[0] = {};
  auto arena = megacu::runtime::task_arena_view{
      .tasks = tasks,
      .events = events,
      .deps = deps,
      .regions = regions,
      .task_capacity = 3,
      .event_capacity = 1,
      .dep_capacity = 2,
      .region_capacity = 1};

  construct_seeded_orch<<<2, 2>>>(arena, device_status, construction_status);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::ok);
  assert(*construction_status ==
         megacu::runtime::execution::seeded_orch::construction_succeeded);
  assert(regions[0].epoch == 0);
  assert(regions[0].first_task == 0);
  assert(regions[0].task_count == 3);
  assert(regions[0].first_event == 0);
  assert(regions[0].event_count == 1);
  assert(regions[0].sealed == 1);
  assert(events[0].ref == megacu::runtime::event_tensor_ref{0});
  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == megacu::runtime::operator_slot{3});
  assert(tasks[0].dep_count == 0);
  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(has_event_wait(tasks[1]));
  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == megacu::runtime::operator_slot{4});
  assert(tasks[2].dep_count == 1);
  assert(!has_event_wait(tasks[2]));
  assert(deps[tasks[1].first_dep] == megacu::runtime::task_ref{0});
  assert(deps[tasks[2].first_dep] == megacu::runtime::task_ref{1});

  seeded_runtime_observation *observations = nullptr;
  assert(cudaMallocManaged(&observations, sizeof(*observations) * 4) ==
         cudaSuccess);
  for (int index = 0; index < 4; ++index) {
    observations[index] = {};
  }
  regions[0] = {.sealed = 1};
  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;
  arena.task_count = 0;
  arena.event_count = 0;
  arena.dep_count = 0;
  arena.region_count = 0;
  instantiate_seeded_persistent_runtime<<<2, 2>>>(arena, device_status,
                                                  construction_status,
                                                  observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::ok);
  assert(*construction_status ==
         megacu::runtime::execution::seeded_orch::construction_succeeded);
  for (int index = 0; index < 4; ++index) {
    assert(observations[index].sealed == 1);
    assert(observations[index].task_count == 3);
    assert(observations[index].event_count == 1);
    assert(observations[index].region_count == 1);
  }

  auto task_overflow_arena = arena;
  task_overflow_arena.task_count = 0;
  task_overflow_arena.event_count = 0;
  task_overflow_arena.dep_count = 0;
  task_overflow_arena.region_count = 0;
  task_overflow_arena.task_capacity = 1;
  task_overflow_arena.event_capacity = 0;
  task_overflow_arena.dep_capacity = 0;
  *device_status = {};
  observations[0] = {};
  publish_task_overflow<<<1, 1>>>(task_overflow_arena, device_status,
                                  observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(observations[0].task_count == 1);
  assert(observations[0].event_count == 0);
  assert(observations[0].dep_count == 0);

  for (int index = 0; index < 4; ++index) {
    observations[index] = {};
  }
  regions[0] = {.sealed = 1};
  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;
  task_overflow_arena.task_count = 0;
  task_overflow_arena.event_count = 0;
  task_overflow_arena.dep_count = 0;
  task_overflow_arena.region_count = 0;
  instantiate_seeded_task_overflow_persistent_runtime<<<2, 2>>>(
      task_overflow_arena, device_status, construction_status, observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(*construction_status ==
         megacu::runtime::execution::seeded_orch::construction_failed);
  assert(regions[0].sealed == 0);
  for (int index = 0; index < 4; ++index) {
    assert(observations[index].sealed == 0);
    assert(observations[index].task_count == 0);
    assert(observations[index].event_count == 0);
    assert(observations[index].dep_count == 0);
    assert(observations[index].region_count == 0);
  }

  *marker = 0;
  regions[0] = {.sealed = 1};
  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;
  task_overflow_arena.task_count = 0;
  task_overflow_arena.event_count = 0;
  task_overflow_arena.dep_count = 0;
  task_overflow_arena.region_count = 0;
  instantiate_seeded_failure_block_tile_runtime<<<2, 2>>>(
      task_overflow_arena, device_status, construction_status, marker);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(*construction_status ==
         megacu::runtime::execution::seeded_orch::construction_failed);
  assert(regions[0].sealed == 0);
  assert(*marker == 0);

  auto dep_overflow_arena = arena;
  dep_overflow_arena.task_count = 0;
  dep_overflow_arena.event_count = 0;
  dep_overflow_arena.dep_count = 0;
  dep_overflow_arena.region_count = 0;
  dep_overflow_arena.task_capacity = 2;
  dep_overflow_arena.event_capacity = 0;
  dep_overflow_arena.dep_capacity = 1;
  *device_status = {};
  observations[0] = {};
  publish_dep_overflow<<<1, 1>>>(dep_overflow_arena, device_status,
                                 observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(observations[0].task_count == 1);
  assert(observations[0].dep_count == 0);

  auto event_overflow_arena = arena;
  event_overflow_arena.task_count = 0;
  event_overflow_arena.event_count = 0;
  event_overflow_arena.dep_count = 0;
  event_overflow_arena.region_count = 0;
  event_overflow_arena.event_capacity = 1;
  *device_status = {};
  observations[0] = {};
  publish_event_overflow<<<1, 1>>>(event_overflow_arena, device_status,
                                   observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(observations[0].task_count == 0);
  assert(observations[0].event_count == 1);
  assert(observations[0].dep_count == 0);

  auto region_overflow_arena = arena;
  region_overflow_arena.task_count = 0;
  region_overflow_arena.event_count = 0;
  region_overflow_arena.dep_count = 0;
  region_overflow_arena.region_count = 0;
  region_overflow_arena.region_capacity = 0;
  *device_status = {};
  observations[0] = {};
  publish_region_overflow<<<1, 1>>>(region_overflow_arena, device_status,
                                    observations);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(device_status->code == megacu::status_code::invalid_argument);
  assert(observations[0].task_count == 1);
  assert(observations[0].event_count == 1);
  assert(observations[0].dep_count == 0);
  assert(observations[0].region_count == 0);

  assert(cudaFree(observations) == cudaSuccess);
  assert(cudaFree(construction_status) == cudaSuccess);
  assert(cudaFree(device_status) == cudaSuccess);
  assert(cudaFree(regions) == cudaSuccess);
  assert(cudaFree(deps) == cudaSuccess);
  assert(cudaFree(events) == cudaSuccess);
  assert(cudaFree(tasks) == cudaSuccess);
  assert(cudaFree(marker) == cudaSuccess);
  return 0;
}
