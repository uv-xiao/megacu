#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

#include <array>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/task_arena.h>

namespace gemm_ar {

struct host_arena_storage {
  std::array<megacu::runtime::device_task_record, 4> tasks{};
  std::array<megacu::runtime::device_event_tensor_record, 2> events{};
  std::array<megacu::runtime::task_ref, 4> deps{};
  std::array<megacu::runtime::arena_region, 1> regions{};
  std::array<std::uint32_t, 4> completed{};
  std::array<std::uint32_t, 4> remaining{};
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

inline void initialize_remaining_work(host_arena_storage &storage,
                                      runtime_args args) {
  auto const tiles = static_cast<std::uint32_t>(
      tile_rows(args.problem) * tile_cols(args.problem));
  for (std::uint32_t task = 0; task < storage.task_count; ++task) {
    storage.completed[task] = 0;
    storage.remaining[task] =
        storage.tasks[task].kind == megacu::runtime::task_kind::sync_only ? 1
                                                                          : tiles;
  }
}

inline megacu::status build_host_arena(host_arena_storage &storage,
                                       runtime_args args) {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
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
