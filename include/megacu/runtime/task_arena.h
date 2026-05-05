#pragma once

#include <cstdint>

#include <megacu/runtime.h>

namespace megacu::runtime {

struct operator_slot {
  std::uint16_t value = 0;
};

constexpr operator_slot invalid_operator_slot() {
  return {static_cast<std::uint16_t>(~std::uint16_t{0})};
}

constexpr bool operator==(operator_slot lhs, operator_slot rhs) {
  return lhs.value == rhs.value;
}

struct device_task_record {
  task_kind kind = task_kind::operator_body;
  operator_slot op_slot = invalid_operator_slot();
  std::uint16_t dep_count = 0;
  std::uint16_t first_dep = 0;
  attr_set attributes;
};

struct device_event_tensor_record {
  event_tensor_ref ref = invalid_event_tensor_ref();
  attr_set attributes;
};

struct arena_region {
  std::uint32_t epoch = 0;
  std::uint32_t first_task = 0;
  std::uint32_t task_count = 0;
  std::uint32_t first_event = 0;
  std::uint32_t event_count = 0;
  std::uint32_t sealed = 0;
};

struct task_arena_view {
  device_task_record *tasks = nullptr;
  device_event_tensor_record *events = nullptr;
  task_ref *deps = nullptr;
  arena_region *regions = nullptr;
  std::uint32_t task_count = 0;
  std::uint32_t event_count = 0;
  std::uint32_t dep_count = 0;
  std::uint32_t region_count = 0;
  std::uint32_t task_capacity = 0;
  std::uint32_t event_capacity = 0;
  std::uint32_t dep_capacity = 0;
  std::uint32_t region_capacity = 0;
};

} // namespace megacu::runtime
