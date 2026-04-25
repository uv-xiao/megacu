#include <cstddef>
#include <cstdint>

#include <megacu/detail/components.h>

namespace megacu::detail {

namespace {

schedule_action action_for_role(placement_role role) {
  if (role == placement_role::comm) {
    return schedule_action::launch_allreduce_tile_consume;
  }
  return schedule_action::launch_gemm_tile_produce;
}

residency_role residency_for_role(placement_role role) {
  if (role == placement_role::comm) {
    return residency_role::comm;
  }
  return residency_role::compute;
}

void apply_events(
    submission_decl const &submission,
    schedule_entry &entry,
    schedule_section &schedule) {
  for (auto const &event : submission.events) {
    if (event.kind == event_use_kind::release) {
      entry.released_event_slot = event.event_slot;
      continue;
    }

    entry.required_event_slot = event.event_slot;
    entry.wait_mode = event.wait_mode;
    if (event.wait_mode == event_wait_mode::blocking_device_wait) {
      schedule.has_blocking_device_wait = true;
    }
  }
}

}  // namespace

schedule_section build_static_persistent_schedule(
    owned_program_ir const &program,
    dispatch_section const &dispatch,
    target_options options) {
  schedule_section schedule;
  schedule.progress = options.progress;

  for (std::size_t index = 0; index < dispatch.work.size(); ++index) {
    auto const &work = dispatch.work[index];
    if (work.role == placement_role::compute) {
      ++schedule.num_compute_workers;
    } else {
      ++schedule.num_comm_workers;
    }

    schedule_entry entry{
        .order = static_cast<std::uint32_t>(index),
        .dispatch_entry_index = static_cast<std::uint32_t>(index),
        .action = action_for_role(work.role),
        .required_event_slot = invalid_slot,
        .released_event_slot = invalid_slot,
        .phase = static_cast<std::uint16_t>(
            options.progress == progress_model::phased &&
                    work.role == placement_role::comm
                ? 1
                : 0),
        .residency_group = static_cast<std::uint16_t>(
            options.progress == progress_model::co_resident_persistent ? 0
                                                                       : invalid_slot),
        .role = residency_for_role(work.role),
        .wait_mode = event_wait_mode::none};

    if (index < program.submissions.size()) {
      apply_events(program.submissions[index], entry, schedule);
    }
    if (entry.required_event_slot != invalid_slot &&
        entry.wait_mode == event_wait_mode::none) {
      entry.wait_mode = event_wait_mode::phase_satisfied;
    }
    schedule.entries.push_back(entry);
  }

  if (options.progress == progress_model::co_resident_persistent) {
    schedule.residency_groups.push_back({
        .group = 0,
        .min_compute_workers = static_cast<std::uint16_t>(
            schedule.num_compute_workers == 0 ? 1 : schedule.num_compute_workers),
        .min_comm_workers = static_cast<std::uint16_t>(
            schedule.num_comm_workers == 0 ? 1 : schedule.num_comm_workers),
        .all_workers_must_be_launched_together = options.co_resident_guard});
    schedule.residency = {
        .persistent_grid_blocks = static_cast<std::uint16_t>(
            schedule.num_compute_workers + schedule.num_comm_workers),
        .threads_per_block = 256,
        .min_sm_count = 1,
        .max_blocks_per_sm = 2,
        .requires_cooperative_launch = false};
  }

  return schedule;
}

}  // namespace megacu::detail

extern "C" int megacu_scheduler_static_persistent_component() {
  return 1;
}
