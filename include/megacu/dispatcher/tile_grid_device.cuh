#pragma once

#include <cstdint>

#include <megacu/runtime/device_types.cuh>

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
};

} // namespace megacu::dispatcher::device
