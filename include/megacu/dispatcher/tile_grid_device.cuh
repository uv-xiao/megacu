#pragma once

#include <cstdint>

#include <megacu/runtime/device_types.cuh>
#include <megacu/runtime/task_arena.h>

namespace megacu::dispatcher::device {

struct tile_grid {
  std::int64_t tiles = 0;

  template <class Context>
  __device__ megacu::runtime::device::work_item first(Context ctx) const {
    auto tile_id = static_cast<std::int64_t>(ctx.block_id());
    return {.active = tile_id < tiles, .tile_id = tile_id};
  }

  template <class Context>
  __device__ megacu::runtime::device::work_item
  next(Context ctx, megacu::runtime::device::work_item current) const {
    auto tile_id = current.tile_id + ctx.grid_blocks();
    return {.active = tile_id < tiles, .tile_id = tile_id};
  }

  template <class Context, class Arena>
  __device__ megacu::runtime::device::work_item
  first(Context ctx, Arena arena,
        megacu::runtime::device::task_ref task) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return {};
    }

    auto const &record = arena.tasks[task.value];
    if (record.kind == megacu::runtime::task_kind::sync_only) {
      return {.active = ctx.block_id() == 0, .tile_id = 0};
    }

    for (auto attr : record.attributes.entries()) {
      if (attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid) {
        auto const tile_count = attr.first * attr.second;
        auto const tile_id = static_cast<std::int64_t>(ctx.block_id());
        return {.active = tile_count > 0 && tile_id < tile_count,
                .tile_id = tile_id};
      }
    }
    return {};
  }

  template <class Context, class Arena>
  __device__ megacu::runtime::device::work_item
  next(Context ctx, Arena arena, megacu::runtime::device::task_ref task,
       megacu::runtime::device::work_item current) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return {};
    }

    auto const &record = arena.tasks[task.value];
    if (record.kind == megacu::runtime::task_kind::sync_only) {
      return {};
    }

    for (auto attr : record.attributes.entries()) {
      if (attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid) {
        auto const tile_count = attr.first * attr.second;
        auto const tile_id = current.tile_id + ctx.grid_blocks();
        return {.active = tile_count > 0 && tile_id < tile_count,
                .tile_id = tile_id};
      }
    }
    return {};
  }
};

} // namespace megacu::dispatcher::device
