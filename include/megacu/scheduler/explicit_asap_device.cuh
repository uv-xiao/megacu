#pragma once

#include <megacu/runtime/device_types.cuh>
#include <megacu/runtime/task_arena.h>

namespace megacu::scheduler::device {

struct explicit_asap {
  int task_count = 0;

  template <class Context, class Work>
  __device__ megacu::runtime::device::task_ref first(Context, Work) const {
    if (task_count <= 0) {
      return {};
    }
    return {.value = 0};
  }

  template <class Context, class Work>
  __device__ megacu::runtime::device::task_ref
  next(Context, Work, megacu::runtime::device::task_ref current) const {
    auto next_task = current.value + 1;
    if (next_task >= task_count) {
      return {};
    }
    return {.value = next_task};
  }

  template <class Context, class Work>
  __device__ void complete(Context, megacu::runtime::device::task_ref,
                           Work) const {}

  template <class Context, class Arena, class Work>
  __device__ megacu::runtime::device::task_ref first(Context, Arena arena,
                                                     Work) const {
    return find_ready(arena, region_first_task(arena));
  }

  template <class Context, class Arena, class Work>
  __device__ megacu::runtime::device::task_ref
  next(Context, Arena arena, Work,
       megacu::runtime::device::task_ref current) const {
    return find_ready(arena, current.value + 1);
  }

  template <class Context, class Arena, class Work>
  __device__ void complete(Context, Arena arena,
                           megacu::runtime::device::task_ref task, Work) const {
    if (!task.valid() || arena.task_completed == nullptr ||
        arena.task_remaining_work == nullptr) {
      return;
    }

    auto const remaining_before =
        atomicSub(&arena.task_remaining_work[task.value], 1);
    if (remaining_before == 1) {
      arena.task_completed[task.value] = 1;
    }
  }

private:
  template <class Arena> __device__ int region_first_task(Arena arena) const {
    if (arena.regions == nullptr || arena.region_count == 0 ||
        arena.regions[0].sealed == 0) {
      return static_cast<int>(arena.task_count);
    }
    return static_cast<int>(arena.regions[0].first_task);
  }

  template <class Arena>
  __device__ megacu::runtime::device::task_ref find_ready(Arena arena,
                                                          int start) const {
    if (arena.tasks == nullptr || arena.regions == nullptr ||
        arena.region_count == 0 || arena.regions[0].sealed == 0) {
      return {};
    }

    auto const first_task = static_cast<int>(arena.regions[0].first_task);
    auto const one_past_last =
        first_task + static_cast<int>(arena.regions[0].task_count);
    for (auto task = start < first_task ? first_task : start;
         task < one_past_last && task < static_cast<int>(arena.task_count);
         ++task) {
      if (is_ready(arena, task)) {
        return {.value = task};
      }
    }
    return {};
  }

  template <class Arena>
  __device__ bool is_ready(Arena arena, int task_index) const {
    if (arena.task_completed != nullptr &&
        arena.task_completed[task_index] != 0) {
      return false;
    }

    auto const &task = arena.tasks[task_index];
    for (std::uint16_t offset = 0; offset < task.dep_count; ++offset) {
      auto const dep_index = task.first_dep + offset;
      if (arena.deps == nullptr || dep_index >= arena.dep_count) {
        return false;
      }
      auto const dep = arena.deps[dep_index];
      if (dep.value >= arena.task_count || arena.task_completed == nullptr ||
          arena.task_completed[dep.value] == 0) {
        return false;
      }
    }
    return true;
  }
};

} // namespace megacu::scheduler::device
