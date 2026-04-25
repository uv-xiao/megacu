#pragma once

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

target_metadata lower_target(owned_program_ir const &program, target_options options);

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
