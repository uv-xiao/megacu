#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

#include <array>
#include <cstddef>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/task_arena.h>

namespace gemm_ar {

inline constexpr std::size_t kMaxTiles = 16;
inline constexpr std::size_t kTaskCapacity = kMaxTiles * 3;
inline constexpr std::size_t kEventCapacity = 2;
inline constexpr std::size_t kDepCapacity = kMaxTiles * 2;
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

  megacu::runtime::task_arena_view view() {
    return {.tasks = tasks.data(),
            .events = events.data(),
            .deps = deps.data(),
            .regions = regions.data(),
            .task_completed = completed.data(),
            .task_remaining_work = remaining.data(),
            .task_count = task_count,
            .event_count = event_count,
            .dep_count = dep_count,
            .region_count = region_count,
            .task_capacity = static_cast<std::uint32_t>(tasks.size()),
            .event_capacity = static_cast<std::uint32_t>(events.size()),
            .dep_capacity = static_cast<std::uint32_t>(deps.size()),
            .region_capacity = static_cast<std::uint32_t>(regions.size())};
  }
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

inline void initialize_remaining_work(host_arena_storage &storage,
                                      runtime_args /*args*/) {
  for (std::uint32_t task = 0; task < storage.task_count; ++task) {
    storage.completed[task] = 0;
    storage.remaining[task] = initial_remaining_work(storage.tasks[task]);
  }
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

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();
  storage.task_count = static_cast<std::uint32_t>(tasks.size());
  storage.event_count = static_cast<std::uint32_t>(events.size());
  storage.dep_count = static_cast<std::uint32_t>(deps.size());
  storage.region_count = static_cast<std::uint32_t>(regions.size());
  for (std::uint32_t index = 0; index < storage.task_count; ++index) {
    storage.tasks[index] = tasks[index];
  }
  for (std::uint32_t index = 0; index < storage.event_count; ++index) {
    storage.events[index] = events[index];
  }
  for (std::uint32_t index = 0; index < storage.dep_count; ++index) {
    storage.deps[index] = deps[index];
  }
  for (std::uint32_t index = 0; index < storage.region_count; ++index) {
    storage.regions[index] = regions[index];
  }
  initialize_remaining_work(storage, args);
  return {};
}

} // namespace gemm_ar
