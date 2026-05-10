#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include <megacu/runtime/device_task_arena.cuh>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

namespace {

constexpr std::uint32_t kTaskCapacity = 16;
constexpr std::uint32_t kEventCapacity = 2;
constexpr std::uint32_t kDepCapacity = 16;
constexpr std::uint32_t kRegionCapacity = 1;

__global__ void build_recipe_on_device(megacu::runtime::task_arena_view arena,
                                       megacu::status *device_status) {
  megacu::runtime::device_orch orch{arena};
  auto args = gemm_ar::runtime_args{
      .driver = {.team = {.team_n_pes = 1}},
      .problem = {.m = 2, .n = 2, .k = 2, .tile_m = 1, .tile_n = 1}};
  auto status = gemm_ar::build_runtime_recipe(orch, args);
  if (status.code == megacu::status_code::ok) {
    status = orch.seal();
  }
  *device_status = status;
}

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

bool has_single_tile(megacu::runtime::attr_set const &attrs,
                     std::int64_t tile_id) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dispatch_single_tile &&
        attr.first == tile_id) {
      return true;
    }
  }
  return false;
}

bool has_wait_tile(megacu::runtime::attr_set const &attrs,
                   megacu::runtime::event_tensor_ref event,
                   std::int64_t tile_id) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_wait &&
        attr.event == event && attr.first == tile_id && attr.second == 1) {
      return true;
    }
  }
  return false;
}

bool has_notify(megacu::runtime::attr_set const &attrs,
                megacu::runtime::event_tensor_ref event) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_notify &&
        attr.event == event) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  megacu::runtime::device_task_record *tasks = nullptr;
  megacu::runtime::device_event_tensor_record *events = nullptr;
  megacu::runtime::task_ref *deps = nullptr;
  megacu::runtime::arena_region *regions = nullptr;
  megacu::status *device_status = nullptr;

  require_cuda(cudaMallocManaged(&tasks, sizeof(*tasks) * kTaskCapacity));
  require_cuda(cudaMallocManaged(&events, sizeof(*events) * kEventCapacity));
  require_cuda(cudaMallocManaged(&deps, sizeof(*deps) * kDepCapacity));
  require_cuda(cudaMallocManaged(&regions, sizeof(*regions) * kRegionCapacity));
  require_cuda(cudaMallocManaged(&device_status, sizeof(*device_status)));

  *device_status = {};
  auto arena = megacu::runtime::task_arena_view{
      .tasks = tasks,
      .events = events,
      .deps = deps,
      .regions = regions,
      .task_count = 0,
      .event_count = 0,
      .dep_count = 0,
      .region_count = 0,
      .task_capacity = kTaskCapacity,
      .event_capacity = kEventCapacity,
      .dep_capacity = kDepCapacity,
      .region_capacity = kRegionCapacity};

  build_recipe_on_device<<<1, 1>>>(arena, device_status);
  require_cuda(cudaDeviceSynchronize());

  assert(device_status->code == megacu::status_code::ok);
  assert(regions[0].sealed == 1);
  assert(regions[0].task_count == 12);
  assert(regions[0].event_count == 1);
  assert(events[0].ref == megacu::runtime::event_tensor_ref{0});

  bool event_has_shape = false;
  bool event_has_wait_count = false;
  bool event_has_storage = false;
  bool event_has_scope = false;
  for (auto attr : events[0].attributes.entries()) {
    event_has_shape |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_shape &&
        attr.first == 2 && attr.second == 2;
    event_has_wait_count |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_wait_count &&
        attr.first == 1;
    event_has_storage |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_storage &&
        attr.pointer == nullptr;
    event_has_scope |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_scope;
  }
  assert(event_has_shape && event_has_wait_count && event_has_storage &&
         event_has_scope);

  for (std::int64_t tile = 0; tile < 4; ++tile) {
    auto const producer_index = static_cast<std::uint32_t>(tile * 3);
    auto const sync_index = producer_index + 1;
    auto const consumer_index = producer_index + 2;
    auto const &producer = tasks[producer_index];
    auto const &sync = tasks[sync_index];
    auto const &consumer = tasks[consumer_index];

    assert(producer.kind == megacu::runtime::task_kind::operator_body);
    assert(producer.op_slot == gemm_ar::runtime_slots::gemm_tile_produce);
    assert(producer.dep_count == 0);
    assert(has_single_tile(producer.attributes, tile));
    assert(has_notify(producer.attributes, events[0].ref));

    assert(sync.kind == megacu::runtime::task_kind::sync_only);
    assert(sync.op_slot == megacu::runtime::invalid_operator_slot());
    assert(sync.dep_count == 1);
    assert(deps[sync.first_dep].value == producer_index);
    assert(has_wait_tile(sync.attributes, events[0].ref, tile));

    assert(consumer.kind == megacu::runtime::task_kind::operator_body);
    assert(consumer.op_slot ==
           gemm_ar::runtime_slots::allreduce_tile_consume);
    assert(consumer.dep_count == 1);
    assert(deps[consumer.first_dep].value == sync_index);
    assert(has_single_tile(consumer.attributes, tile));
  }
  assert(tasks[11].first_dep + tasks[11].dep_count == 8);

  require_cuda(cudaFree(device_status));
  require_cuda(cudaFree(regions));
  require_cuda(cudaFree(deps));
  require_cuda(cudaFree(events));
  require_cuda(cudaFree(tasks));
  return 0;
}
