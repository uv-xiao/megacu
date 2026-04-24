#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include <megacu/detail/target_metadata.h>
#include <megacu/program.h>

namespace megacu::detail {

struct owned_program_ir {
  std::vector<extent_decl> extents;
  std::vector<std::vector<slot_index>> domain_extent_storage;
  std::vector<domain_decl> domains;
  std::vector<participant_decl> participants;
  std::vector<resource_decl> resources;
  std::vector<std::vector<slot_index>> event_domain_storage;
  std::vector<event_decl> events;
  std::vector<std::vector<arg_binding>> submission_arg_storage;
  std::vector<std::vector<event_use>> submission_event_storage;
  std::vector<submission_decl> submissions;
};

struct materialized_target {
  megacu::status status;
  owned_program_ir program;
  target_metadata metadata;
};

inline owned_program_ir own(program_ir view) {
  owned_program_ir owned;
  owned.extents.assign(view.extents.begin(), view.extents.end());
  owned.participants.assign(view.participants.begin(), view.participants.end());
  owned.resources.assign(view.resources.begin(), view.resources.end());

  for (auto domain : view.domains) {
    owned.domain_extent_storage.emplace_back(
        domain.extent_slots.begin(), domain.extent_slots.end());
    domain.extent_slots = owned.domain_extent_storage.back();
    owned.domains.push_back(domain);
  }

  for (auto event : view.events) {
    owned.event_domain_storage.emplace_back(
        event.domain_slots.begin(), event.domain_slots.end());
    event.domain_slots = owned.event_domain_storage.back();
    owned.events.push_back(event);
  }

  for (auto submission : view.submissions) {
    owned.submission_arg_storage.emplace_back(
        submission.args.begin(), submission.args.end());
    owned.submission_event_storage.emplace_back(
        submission.events.begin(), submission.events.end());
    submission.args = owned.submission_arg_storage.back();
    submission.events = owned.submission_event_storage.back();
    owned.submissions.push_back(submission);
  }

  return owned;
}

inline bool has_blocking_wait(owned_program_ir const &program) {
  for (auto const &submission : program.submissions) {
    for (auto const &event : submission.events) {
      if (event.wait_mode == event_wait_mode::blocking_device_wait) {
        return true;
      }
    }
  }
  return false;
}

inline target_metadata lower_target(
    owned_program_ir const &program,
    target_options options) {
  target_metadata metadata;
  metadata.target_name = std::string(options.target_name);
  metadata.program.extent_count = static_cast<std::uint16_t>(program.extents.size());
  metadata.program.domain_count = static_cast<std::uint16_t>(program.domains.size());
  metadata.program.participant_count =
      static_cast<std::uint16_t>(program.participants.size());
  metadata.program.resource_count =
      static_cast<std::uint16_t>(program.resources.size());
  metadata.program.event_count = static_cast<std::uint16_t>(program.events.size());
  metadata.program.submission_count =
      static_cast<std::uint16_t>(program.submissions.size());

  metadata.dispatch.work_entry_count =
      static_cast<std::uint16_t>(program.submissions.size());
  metadata.dispatch.participant_entry_count =
      static_cast<std::uint16_t>(program.participants.size() * options.team_size);

  metadata.schedule.progress = options.progress;
  metadata.schedule.has_blocking_device_wait = has_blocking_wait(program);
  if (options.progress == progress_model::co_resident_persistent) {
    metadata.schedule.residency_groups.push_back({
        .group = 0,
        .min_compute_workers = 1,
        .min_comm_workers = 1,
        .all_workers_must_be_launched_together = options.co_resident_guard});
  }

  for (auto const &submission : program.submissions) {
    auto found = std::find_if(
        metadata.kernel.symbols.begin(),
        metadata.kernel.symbols.end(),
        [&](kernel_symbol const &symbol) {
          return symbol.op_name == submission.op_name;
        });
    if (found == metadata.kernel.symbols.end()) {
      metadata.kernel.symbols.push_back({
          .op_name = std::string(submission.op_name),
          .op_slot = submission.op_slot});
    }
  }
  metadata.kernel.stitched_persistent =
      options.progress == progress_model::co_resident_persistent;

  metadata.backend.team_size = options.team_size;
  metadata.backend.event_storage_bytes =
      static_cast<std::uint32_t>(program.events.size() * options.team_size *
                                 sizeof(std::uint64_t));
  metadata.backend.requires_symmetric_partial_buffer = true;
  metadata.backend.may_use_multimem_reduce = true;

  return metadata;
}

template <class Program>
materialized_target materialize_program(target_options options) {
  program_builder builder;
  Program::describe(builder);

  materialized_target target;
  target.program = own(builder.ir());

  auto blocking_wait = has_blocking_wait(target.program);
  if (options.progress == progress_model::co_resident_persistent &&
      blocking_wait && !options.co_resident_guard) {
    target.status = {
        status_code::unsupported,
        1,
        "blocking wait requires co-resident progress guard"};
    return target;
  }

  target.metadata = lower_target(target.program, options);
  target.status = {};
  return target;
}

}  // namespace megacu::detail
