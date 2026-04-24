#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <typeindex>

namespace megacu {

enum class memory_scope : std::uint8_t {
  local_device,
  remote_team
};

enum class event_wait_mode : std::uint8_t {
  none,
  blocking_device_wait,
  phase_satisfied
};

}  // namespace megacu

namespace megacu::detail {

using slot_index = std::uint16_t;
inline constexpr slot_index invalid_slot = static_cast<slot_index>(~slot_index{0});

enum class extent_source : std::uint8_t {
  runtime_problem_field,
  backend_team_size,
  constant
};

struct extent_decl {
  slot_index slot = invalid_slot;
  std::string_view label;
  extent_source source = extent_source::runtime_problem_field;
  std::string_view source_name;
};

struct domain_decl {
  slot_index slot = invalid_slot;
  std::string_view label;
  std::span<const slot_index> extent_slots;
};

struct participant_decl {
  slot_index slot = invalid_slot;
  std::string_view label;
};

struct resource_decl {
  slot_index slot = invalid_slot;
  std::string_view label;
  std::type_index view_type = typeid(void);
};

struct event_decl {
  slot_index slot = invalid_slot;
  std::string_view label;
  std::span<const slot_index> domain_slots;
  slot_index from_participant = invalid_slot;
  slot_index to_participant = invalid_slot;
  slot_index storage_resource = invalid_slot;
  megacu::memory_scope scope = megacu::memory_scope::local_device;
};

enum class event_use_kind : std::uint8_t {
  acquire_one,
  acquire_all,
  release
};

struct event_use {
  slot_index event_slot = invalid_slot;
  event_use_kind kind = event_use_kind::release;
  slot_index over_domain_slot = invalid_slot;
  megacu::event_wait_mode wait_mode = megacu::event_wait_mode::none;
};

struct arg_binding {
  std::string_view name;
  slot_index resource_slot = invalid_slot;
  std::string_view member_name;
};

struct submission_decl {
  slot_index op_slot = invalid_slot;
  std::string_view op_name;
  slot_index work_domain_slot = invalid_slot;
  slot_index participant_slot = invalid_slot;
  std::uint16_t phase = 0;
  std::span<const arg_binding> args;
  std::span<const event_use> events;
};

struct program_ir {
  std::span<const extent_decl> extents;
  std::span<const domain_decl> domains;
  std::span<const participant_decl> participants;
  std::span<const resource_decl> resources;
  std::span<const event_decl> events;
  std::span<const submission_decl> submissions;
};

}  // namespace megacu::detail
