#include <megacu/detail/runtime_validation.h>

namespace megacu::detail {

namespace {

bool same_backend_session(
    megacu::symmetric_buffer_view lhs,
    megacu::symmetric_buffer_view rhs) {
  return lhs.backend.value == rhs.backend.value &&
         lhs.session.value == rhs.session.value;
}

}  // namespace

megacu::status validate_nvshmem_team(
    megacu::nvshmem::team_view team,
    std::int32_t max_supported_team_size) {
  if (team.team_n_pes <= 0 || team.team_n_pes > max_supported_team_size) {
    return {megacu::status_code::invalid_argument, 2, "invalid team envelope"};
  }
  if (team.team_my_pe < 0 || team.team_my_pe >= team.team_n_pes ||
      team.world_n_pes < team.team_n_pes) {
    return {megacu::status_code::invalid_argument, 2, "invalid team envelope"};
  }
  return {};
}

megacu::status validate_nvshmem_symmetric_storage(
    megacu::event_storage_view events,
    megacu::symmetric_tensor_view partial,
    megacu::nvshmem::team_view team,
    std::int64_t required_event_bytes) {
  if (events.buffer.data == nullptr ||
      events.buffer.bytes < required_event_bytes) {
    return {megacu::status_code::invalid_argument, 5, "invalid event storage"};
  }

  megacu::symmetric_buffer_view team_identity{
      .backend = team.backend,
      .session = team.session};
  if (!same_backend_session(events.buffer, team_identity) ||
      !same_backend_session(partial.buffer, team_identity)) {
    return {
        megacu::status_code::invalid_argument,
        6,
        "symmetric storage session mismatch"};
  }

  return {};
}

}  // namespace megacu::detail
