#include "examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_arena.cuh"

#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>

namespace tiny_decode_runtime {
namespace {

std::uint32_t initial_remaining_work(megacu::runtime::device_task_record task) {
  if (task.kind == megacu::runtime::task_kind::sync_only) {
    return 1;
  }
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid) {
      return static_cast<std::uint32_t>(attr.first * attr.second);
    }
  }
  return 1;
}

} // namespace

megacu::status build_host_arena(host_arena_storage &storage,
                                runtime_args args) {
  megacu::runtime::execution::host_orch::frame<task_capacity, event_capacity,
                                               dep_capacity>
      frame;
  auto status = build_recipe(frame, args);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  status = frame.seal();
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  storage.task_count = static_cast<std::uint32_t>(frame.tasks().size());
  storage.event_count =
      static_cast<std::uint32_t>(frame.event_tensors().size());
  storage.dep_count = static_cast<std::uint32_t>(frame.deps().size());
  storage.region_count = static_cast<std::uint32_t>(frame.regions().size());
  for (std::uint32_t index = 0; index < storage.task_count; ++index) {
    storage.tasks[index] = frame.tasks()[index];
    storage.completed[index] = 0;
    storage.remaining[index] = initial_remaining_work(storage.tasks[index]);
  }
  for (std::uint32_t index = 0; index < storage.event_count; ++index) {
    storage.events[index] = frame.event_tensors()[index];
  }
  for (std::uint32_t index = 0; index < storage.dep_count; ++index) {
    storage.deps[index] = frame.deps()[index];
  }
  for (std::uint32_t index = 0; index < storage.region_count; ++index) {
    storage.regions[index] = frame.regions()[index];
  }
  return {};
}

} // namespace tiny_decode_runtime
