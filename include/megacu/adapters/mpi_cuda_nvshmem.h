#pragma once

#include <cstdint>

#include <megacu/runtime.h>

namespace megacu::adapters::mpi_cuda_nvshmem {

struct launch_facts {
  std::int32_t rank = 0;
  std::int32_t world_size = 1;
  std::int32_t local_rank = 0;
};

inline megacu::cuda_nvshmem::driver_view make_driver(
    launch_facts facts,
    void *stream = nullptr) {
  return {
      .launch = {.stream = stream, .device_ordinal = facts.local_rank},
      .team = {.team_my_pe = facts.rank,
               .team_n_pes = facts.world_size,
               .world_my_pe = facts.rank,
               .world_n_pes = facts.world_size,
               .cuda_device_ordinal = facts.local_rank},
  };
}

} // namespace megacu::adapters::mpi_cuda_nvshmem
