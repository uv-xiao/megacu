#pragma once

#include <cstdint>

namespace megacu::runtime::device {

struct task_ref {
  int value = -1;

  __device__ bool valid() const { return value >= 0; }
};

struct work_item {
  bool active = false;
  std::int64_t tile_id = 0;
};

} // namespace megacu::runtime::device
