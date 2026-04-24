#pragma once

#include <cstddef>
#include <string_view>
#include <typeindex>
#include <utility>
#include <vector>

#include <megacu/detail/program_ir.h>
#include <megacu/views.h>

namespace megacu {

namespace detail {

struct extent_handle {
  slot_index slot = invalid_slot;
};

struct domain_handle {
  slot_index slot = invalid_slot;
};

struct participant_handle {
  slot_index slot = invalid_slot;
};

struct resource_handle {
  slot_index slot = invalid_slot;
};

}  // namespace detail

template <class Tag>
struct extent_ref {
  detail::slot_index slot = detail::invalid_slot;

  constexpr operator detail::extent_handle() const { return {slot}; }
};

template <class Tag>
struct domain_ref {
  detail::slot_index slot = detail::invalid_slot;

  constexpr operator detail::domain_handle() const { return {slot}; }
};

template <class Tag>
struct participant_ref {
  detail::slot_index slot = detail::invalid_slot;

  constexpr operator detail::participant_handle() const { return {slot}; }
};

template <class SlotTag, class View>
struct resource_ref : View {
  detail::slot_index slot = detail::invalid_slot;

  constexpr operator detail::resource_handle() const { return {slot}; }
};

struct domain_set {
  std::vector<detail::slot_index> slots;
};

template <class... Domains>
domain_set over(Domains const &...domains) {
  domain_set result;
  result.slots.reserve(sizeof...(Domains));
  (result.slots.push_back(static_cast<detail::domain_handle>(domains).slot), ...);
  return result;
}

struct placement {
  detail::slot_index participant_slot = detail::invalid_slot;
};

template <class Participant>
placement place(Participant const &participant) {
  return {static_cast<detail::participant_handle>(participant).slot};
}

struct remote_event {
  detail::participant_handle from;
  detail::participant_handle to;
  detail::resource_handle storage;
  memory_scope scope = memory_scope::local_device;
};

namespace event_wait {

struct spec {
  event_wait_mode mode = event_wait_mode::none;
};

constexpr spec blocking_device() {
  return {event_wait_mode::blocking_device_wait};
}

constexpr spec phase_satisfied() {
  return {event_wait_mode::phase_satisfied};
}

}  // namespace event_wait

template <class Tag>
struct event_ref {
  detail::slot_index slot = detail::invalid_slot;

  constexpr detail::event_use release() const {
    return {
        .event_slot = slot,
        .kind = detail::event_use_kind::release,
        .over_domain_slot = detail::invalid_slot,
        .wait_mode = event_wait_mode::none};
  }

  template <class Domain>
  constexpr detail::event_use acquire_all(
      Domain const &domain,
      event_wait::spec wait = event_wait::phase_satisfied()) const {
    return {
        .event_slot = slot,
        .kind = detail::event_use_kind::acquire_all,
        .over_domain_slot = static_cast<detail::domain_handle>(domain).slot,
        .wait_mode = wait.mode};
  }
};

struct arg_pack {
  std::vector<detail::arg_binding> args;
  std::vector<detail::event_use> events;

  template <class T>
  arg_pack a(T const &) && {
    args.push_back({.name = "a", .member_name = "a"});
    return std::move(*this);
  }

  template <class T>
  arg_pack b(T const &) && {
    args.push_back({.name = "b", .member_name = "b"});
    return std::move(*this);
  }

  template <class T>
  arg_pack partial(T const &) && {
    args.push_back({.name = "partial", .member_name = "partial"});
    return std::move(*this);
  }

  template <class T>
  arg_pack out(T const &) && {
    args.push_back({.name = "out", .member_name = "out"});
    return std::move(*this);
  }

  arg_pack release(detail::event_use use) && {
    events.push_back(use);
    return std::move(*this);
  }

  arg_pack acquire(detail::event_use use) && {
    events.push_back(use);
    return std::move(*this);
  }
};

inline arg_pack args() {
  return {};
}

class program_builder {
 public:
  template <class Tag>
  extent_ref<Tag> extent(std::string_view label) {
    return add_extent<Tag>(
        label, detail::extent_source::runtime_problem_field, label);
  }

  template <class Tag>
  extent_ref<Tag> backend_extent(std::string_view label) {
    return add_extent<Tag>(label, detail::extent_source::backend_team_size, label);
  }

  template <class Tag>
  extent_ref<Tag> constant_extent(std::string_view label) {
    return add_extent<Tag>(label, detail::extent_source::constant, label);
  }

  extent_ref<void> backend_extent(std::string_view label) {
    return add_extent<void>(
        label, detail::extent_source::backend_team_size, label);
  }

  template <class Tag, class... Extents>
  domain_ref<Tag> domain(std::string_view label, Extents const &...extents) {
    std::vector<detail::slot_index> extent_slots;
    extent_slots.reserve(sizeof...(Extents));
    (extent_slots.push_back(static_cast<detail::extent_handle>(extents).slot), ...);

    auto slot = next_slot(domains_.size());
    domain_extent_storage_.push_back(std::move(extent_slots));
    domains_.push_back({
        .slot = slot,
        .label = label,
        .extent_slots = domain_extent_storage_.back()});
    return {slot};
  }

  template <class Tag>
  participant_ref<Tag> participant(std::string_view label) {
    auto slot = next_slot(participants_.size());
    participants_.push_back({.slot = slot, .label = label});
    return {slot};
  }

  template <class SlotTag, class View>
  resource_ref<SlotTag, View> resource(std::string_view label) {
    auto slot = next_slot(resources_.size());
    resources_.push_back({
        .slot = slot,
        .label = label,
        .view_type = std::type_index(typeid(View))});
    resource_ref<SlotTag, View> result{};
    result.slot = slot;
    return result;
  }

  template <class Tag>
  event_ref<Tag> event(
      std::string_view label,
      domain_set domains,
      remote_event event) {
    auto slot = next_slot(events_.size());
    event_domain_storage_.push_back(std::move(domains.slots));
    events_.push_back({
        .slot = slot,
        .label = label,
        .domain_slots = event_domain_storage_.back(),
        .from_participant = event.from.slot,
        .to_participant = event.to.slot,
        .storage_resource = event.storage.slot,
        .scope = event.scope});
    return {slot};
  }

  template <class Op>
  void submit(Op, domain_set work, placement where, arg_pack pack) {
    auto slot = next_slot(submissions_.size());
    submission_arg_storage_.push_back(std::move(pack.args));
    submission_event_storage_.push_back(std::move(pack.events));
    submissions_.push_back({
        .op_slot = slot,
        .op_name = Op::name,
        .work_domain_slot = work.slots.empty() ? detail::invalid_slot : work.slots[0],
        .participant_slot = where.participant_slot,
        .phase = 0,
        .args = submission_arg_storage_.back(),
        .events = submission_event_storage_.back()});
  }

  [[nodiscard]] detail::program_ir ir() const {
    return {
        .extents = extents_,
        .domains = domains_,
        .participants = participants_,
        .resources = resources_,
        .events = events_,
        .submissions = submissions_};
  }

 private:
  static detail::slot_index next_slot(std::size_t size) {
    return static_cast<detail::slot_index>(size);
  }

  template <class Tag>
  extent_ref<Tag> add_extent(
      std::string_view label,
      detail::extent_source source,
      std::string_view source_name) {
    auto slot = next_slot(extents_.size());
    extents_.push_back({
        .slot = slot,
        .label = label,
        .source = source,
        .source_name = source_name});
    return {slot};
  }

  std::vector<detail::extent_decl> extents_;
  std::vector<std::vector<detail::slot_index>> domain_extent_storage_;
  std::vector<detail::domain_decl> domains_;
  std::vector<detail::participant_decl> participants_;
  std::vector<detail::resource_decl> resources_;
  std::vector<std::vector<detail::slot_index>> event_domain_storage_;
  std::vector<detail::event_decl> events_;
  std::vector<std::vector<detail::arg_binding>> submission_arg_storage_;
  std::vector<std::vector<detail::event_use>> submission_event_storage_;
  std::vector<detail::submission_decl> submissions_;
};

}  // namespace megacu
