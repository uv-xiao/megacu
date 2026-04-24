#pragma once

#include <cstdint>

#include <megacu/views.h>

namespace megacu::nvshmem {

enum class ownership : std::uint8_t {
  external,
  owned
};

struct team_view {
  void *team = nullptr;
  std::int32_t team_my_pe = 0;
  std::int32_t team_n_pes = 1;
  std::int32_t world_my_pe = 0;
  std::int32_t world_n_pes = 1;
  std::int32_t cuda_device_ordinal = 0;
  megacu::backend_id backend;
  megacu::session_id session;
  ownership owner = ownership::external;
};

}  // namespace megacu::nvshmem
