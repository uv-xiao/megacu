#include <cstdint>

#include <megacu/detail/components.h>

namespace megacu::detail {

namespace {

placement_role role_for_submission(
    owned_program_ir const &program,
    submission_decl const &submission) {
  for (auto const &event : program.events) {
    if (submission.participant_slot == event.from_participant) {
      return placement_role::compute;
    }
    if (submission.participant_slot == event.to_participant) {
      return placement_role::comm;
    }
  }
  return placement_role::compute;
}

}  // namespace

dispatch_section build_tiled_compute_comm_dispatch(
    owned_program_ir const &program,
    target_options options) {
  dispatch_section dispatch;

  std::uint16_t worker_index = 0;
  for (auto const &submission : program.submissions) {
    dispatch.work.push_back({
        .tile = {.m = 0, .n = 0},
        .role = role_for_submission(program, submission),
        .participant_slot = submission.participant_slot,
        .logical_rank = 0,
        .worker_index = worker_index++});
  }

  for (auto const &participant : program.participants) {
    for (std::uint16_t rank = 0; rank < options.team_size; ++rank) {
      dispatch.participants.push_back({
          .tile = {.m = 0, .n = 0},
          .participant_slot = participant.slot,
          .logical_rank = rank,
          .backend_peer = rank});
    }
  }

  dispatch.work_entry_count =
      static_cast<std::uint16_t>(dispatch.work.size());
  dispatch.participant_entry_count =
      static_cast<std::uint16_t>(dispatch.participants.size());
  return dispatch;
}

}  // namespace megacu::detail

extern "C" int megacu_dispatcher_tiled_compute_comm_component() {
  return 1;
}
