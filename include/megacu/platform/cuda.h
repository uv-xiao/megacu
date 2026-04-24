#pragma once

#include <cstdint>

namespace megacu::cuda {

struct launch_view {
  void *stream = nullptr;
  std::int32_t device_ordinal = 0;
};

struct kernel_context {
  void const *target_metadata = nullptr;
  std::uint32_t dispatch_entry_index = 0;
  std::uint16_t local_rank_value = 0;
  std::uint16_t team_size_value = 1;
};

}  // namespace megacu::cuda
