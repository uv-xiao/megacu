#pragma once

#include <cstdint>

#include <megacu/runtime/device_task_arena.cuh>

namespace megacu::runtime::execution::seeded_orch {

inline constexpr std::uint32_t construction_pending = 0;
inline constexpr std::uint32_t construction_succeeded = 1;
inline constexpr std::uint32_t construction_failed = 2;

namespace detail {

__device__ inline void clear_failure_state(task_arena_view &view) {
  view.task_count = 0;
  view.event_count = 0;
  view.dep_count = 0;
  view.region_count = 0;
  if (view.regions != nullptr && view.region_capacity > 0) {
    static_cast<volatile arena_region *>(view.regions)[0].sealed = 0;
  }
}

__device__ inline std::uint32_t published_dep_count(task_arena_view const &view,
                                                    arena_region region) {
  std::uint32_t count = 0;
  auto const one_past_last = region.first_task + region.task_count;
  for (auto task = region.first_task; task < one_past_last; ++task) {
    auto const &record = view.tasks[task];
    auto const task_dep_end =
        static_cast<std::uint32_t>(record.first_dep) + record.dep_count;
    if (task_dep_end > count) {
      count = task_dep_end;
    }
  }
  return count;
}

} // namespace detail

template <class Recipe> struct model {
  task_arena_view arena;
  Recipe recipe;
  status *device_status = nullptr;
  std::uint32_t *construction_status = nullptr;

  template <class Context> __device__ task_arena_view bind(Context) const {
    return arena;
  }

  template <class Context>
  __device__ bool construct(Context ctx, task_arena_view &view) const {
    auto control_writer = ctx.block_id() == 0 && ctx.thread_id() == 0;
    auto handoff =
        static_cast<volatile std::uint32_t *>(construction_status);

    if (control_writer) {
      device_orch orch{view};
      auto result = recipe(orch);
      if (device_status != nullptr) {
        *device_status = result;
      }
      if (result.code != status_code::ok) {
        detail::clear_failure_state(view);
      }
      __threadfence();
      if (handoff != nullptr) {
        handoff[0] = result.code == status_code::ok ? construction_succeeded
                                                     : construction_failed;
      } else {
        return result.code == status_code::ok;
      }
    }

    if (handoff == nullptr) {
      return false;
    }

    while (handoff[0] == construction_pending) {
    }

    if (handoff[0] != construction_succeeded) {
      detail::clear_failure_state(view);
      return false;
    }

    if (view.region_capacity == 0) {
      detail::clear_failure_state(view);
      return false;
    }

    auto volatile_regions = static_cast<volatile arena_region *>(view.regions);
    while (volatile_regions[0].sealed == 0) {
    }

    auto region = view.regions[0];
    view.task_count = region.first_task + region.task_count;
    view.event_count = region.first_event + region.event_count;
    view.dep_count = detail::published_dep_count(view, region);
    view.region_count = 1;
    return true;
  }
};

} // namespace megacu::runtime::execution::seeded_orch
