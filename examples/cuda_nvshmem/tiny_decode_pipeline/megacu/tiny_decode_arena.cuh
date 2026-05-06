#pragma once

#include "examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_runtime_recipe.cuh"

#include <cstdint>

#include <megacu/runtime/task_arena.h>

namespace tiny_decode_runtime {

inline constexpr std::uint32_t task_capacity = 10;
inline constexpr std::uint32_t event_capacity = 5;
inline constexpr std::uint32_t dep_capacity = 12;
inline constexpr std::uint32_t region_capacity = 1;
inline constexpr std::int64_t max_event_tiles = tiny::hidden_size;

struct host_arena_storage {
  megacu::runtime::device_task_record tasks[task_capacity]{};
  megacu::runtime::device_event_tensor_record events[event_capacity]{};
  megacu::runtime::task_ref deps[dep_capacity]{};
  megacu::runtime::arena_region regions[region_capacity]{};
  std::uint32_t completed[task_capacity]{};
  std::uint32_t remaining[task_capacity]{};
  std::uint32_t task_count = 0;
  std::uint32_t event_count = 0;
  std::uint32_t dep_count = 0;
  std::uint32_t region_count = 0;
};

megacu::status build_host_arena(host_arena_storage &storage,
                                runtime_args args);

} // namespace tiny_decode_runtime
