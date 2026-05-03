#include <megacu/backends/nvshmem.h>

namespace megacu::nvshmem {

std::int32_t team_size(team_view team) noexcept {
  return team.team_n_pes;
}

bool has_remote_pes(team_view team) noexcept {
  return team.team_n_pes > 1;
}

}  // namespace megacu::nvshmem

extern "C" int megacu_backend_nvshmem_component() {
  return 1;
}
