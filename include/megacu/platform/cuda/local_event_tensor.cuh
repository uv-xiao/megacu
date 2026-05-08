#pragma once

#include <cstdint>

#include <megacu/platform/cuda.h>
#include <megacu/runtime.h>

namespace megacu::platform::cuda {

struct local_event_tensor_i32 {
  int *events = nullptr;
  std::int64_t max_tiles = 1;

  template <class Context, class Arena, class Task, class Work>
  __device__ bool complete(Context ctx, Arena arena, Task task, Work) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return true;
    }
    if (is_cta_leader(ctx)) {
      for (auto attr : arena.tasks[task.value].attributes.entries()) {
        if (attr.kind == megacu::runtime::attr_kind::event_wait) {
          if (attr.first >= 0) {
            wait_tile(attr.event, attr.first);
          } else {
            wait_event(arena, attr.event);
          }
        }
      }
    }
    return true;
  }

  template <class Context, class Arena, class Task, class Work>
  __device__ void after(Context ctx, Arena arena, Task task, Work work) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return;
    }
    if (is_cta_leader(ctx)) {
      for (auto attr : arena.tasks[task.value].attributes.entries()) {
        if (attr.kind == megacu::runtime::attr_kind::event_notify) {
          notify(attr.event, work.tile_id);
        }
      }
    }
  }

private:
  template <class Context> __device__ bool is_cta_leader(Context ctx) const {
    if constexpr (requires { ctx.thread_id(); }) {
      return ctx.thread_id() == 0;
    } else {
      return true;
    }
  }

  template <class Arena>
  __device__ void wait_event(Arena arena,
                             megacu::runtime::event_tensor_ref event) const {
    auto const tile_count = event_tile_count(arena, event);
    for (std::int64_t tile_id = 0; tile_id < tile_count; ++tile_id) {
      wait_tile(event, tile_id);
    }
  }

  __device__ void wait_tile(megacu::runtime::event_tensor_ref event,
                            std::int64_t tile_id) const {
    megacu::cuda::device::wait_ready(events, slot(event, tile_id),
                                     ready_value(tile_id));
  }

  __device__ void notify(megacu::runtime::event_tensor_ref event,
                         std::int64_t tile_id) const {
    megacu::cuda::device::fence_system();
    megacu::cuda::device::signal_ready(events, slot(event, tile_id),
                                       ready_value(tile_id));
  }

  template <class Arena>
  __device__ std::int64_t
  event_tile_count(Arena arena, megacu::runtime::event_tensor_ref event) const {
    if (arena.events == nullptr ||
        event.value >= static_cast<std::uint32_t>(arena.event_count)) {
      return 1;
    }
    for (auto attr : arena.events[event.value].attributes.entries()) {
      if (attr.kind == megacu::runtime::attr_kind::event_tensor_shape) {
        return attr.first * attr.second;
      }
    }
    return 1;
  }

  __device__ std::int64_t slot(megacu::runtime::event_tensor_ref event,
                               std::int64_t tile_id) const {
    return static_cast<std::int64_t>(event.value) * max_tiles + tile_id;
  }

  __device__ int ready_value(std::int64_t tile_id) const {
    return static_cast<int>(tile_id + 1);
  }
};

} // namespace megacu::platform::cuda
