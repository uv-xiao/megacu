#pragma once

#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm_runtime_recipe.cuh"

#include <array>
#include <cstddef>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/task_arena.h>

namespace ag_gemm {

inline constexpr std::size_t kTaskCapacity = 64;
inline constexpr std::size_t kEventCapacity = 2;
inline constexpr std::size_t kDepCapacity = 128;
inline constexpr std::size_t kRegionCapacity = 1;

struct host_arena_storage {
  std::array<megacu::runtime::device_task_record, kTaskCapacity> tasks{};
  std::array<megacu::runtime::device_event_tensor_record, kEventCapacity>
      events{};
  std::array<megacu::runtime::task_ref, kDepCapacity> deps{};
  std::array<megacu::runtime::arena_region, kRegionCapacity> regions{};
  std::array<std::uint32_t, kTaskCapacity> completed{};
  std::array<std::uint32_t, kTaskCapacity> remaining{};
  std::uint32_t task_count = 0;
  std::uint32_t event_count = 0;
  std::uint32_t dep_count = 0;
  std::uint32_t region_count = 0;
};

inline std::uint32_t initial_remaining_work(
    megacu::runtime::device_task_record task) {
  if (task.kind == megacu::runtime::task_kind::sync_only) {
    return 1;
  }
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dispatch_single_tile) {
      return 1;
    }
    if (attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid) {
      return static_cast<std::uint32_t>(attr.first * attr.second);
    }
  }
  return 1;
}

inline megacu::status build_host_arena(host_arena_storage &storage,
                                       runtime_args args) {
  megacu::runtime::execution::host_orch::frame<kTaskCapacity, kEventCapacity,
                                               kDepCapacity, kRegionCapacity>
      frame;
  auto status = build_runtime_recipe(frame, args);
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

} // namespace ag_gemm
