#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <megacu/detail/program_ir.h>
#include <megacu/views.h>

namespace megacu::detail {

enum class progress_model : std::uint8_t {
  phased,
  co_resident_persistent
};

struct target_options {
  std::string_view target_name;
  progress_model progress = progress_model::phased;
  std::uint16_t team_size = 1;
  bool co_resident_guard = false;
};

struct program_section {
  std::uint16_t extent_count = 0;
  std::uint16_t domain_count = 0;
  std::uint16_t participant_count = 0;
  std::uint16_t resource_count = 0;
  std::uint16_t event_count = 0;
  std::uint16_t submission_count = 0;
};

enum class placement_role : std::uint8_t {
  compute,
  comm
};

struct domain_tile_2d {
  std::uint32_t m = 0;
  std::uint32_t n = 0;
};

struct dispatch_entry {
  domain_tile_2d tile;
  placement_role role = placement_role::compute;
  std::uint16_t participant_slot = invalid_slot;
  std::uint16_t logical_rank = 0;
  std::uint16_t worker_index = 0;
};

struct participant_entry {
  domain_tile_2d tile;
  std::uint16_t participant_slot = invalid_slot;
  std::uint16_t logical_rank = 0;
  std::uint16_t backend_peer = 0;
};

struct dispatch_section {
  std::uint16_t work_entry_count = 0;
  std::uint16_t participant_entry_count = 0;
  std::vector<dispatch_entry> work;
  std::vector<participant_entry> participants;
};

enum class schedule_action : std::uint8_t {
  launch_gemm_tile_produce,
  launch_allreduce_tile_consume
};

using event_wait_mode = megacu::event_wait_mode;

enum class residency_role : std::uint8_t {
  compute,
  comm
};

struct residency_group {
  std::uint16_t group = 0;
  std::uint16_t min_compute_workers = 0;
  std::uint16_t min_comm_workers = 0;
  bool all_workers_must_be_launched_together = false;
};

struct cuda_residency_envelope {
  std::uint16_t persistent_grid_blocks = 0;
  std::uint16_t threads_per_block = 0;
  std::uint16_t min_sm_count = 0;
  std::uint16_t max_blocks_per_sm = 0;
  bool requires_cooperative_launch = false;
};

struct schedule_entry {
  std::uint32_t order = 0;
  std::uint32_t dispatch_entry_index = 0;
  schedule_action action = schedule_action::launch_gemm_tile_produce;
  std::uint16_t required_event_slot = invalid_slot;
  std::uint16_t released_event_slot = invalid_slot;
  std::uint16_t phase = 0;
  std::uint16_t residency_group = invalid_slot;
  residency_role role = residency_role::compute;
  event_wait_mode wait_mode = event_wait_mode::none;
};

struct schedule_section {
  progress_model progress = progress_model::phased;
  std::uint16_t num_compute_workers = 0;
  std::uint16_t num_comm_workers = 0;
  std::vector<residency_group> residency_groups;
  cuda_residency_envelope residency;
  std::vector<schedule_entry> entries;
  bool has_blocking_device_wait = false;
};

enum class entrypoint_role : std::uint8_t {
  launchable_kernel,
  callable_body
};

struct kernel_symbol {
  std::string op_name;
  std::uint16_t op_slot = invalid_slot;
  entrypoint_role role = entrypoint_role::callable_body;
  std::uint16_t symbol_id = invalid_slot;
};

struct cuda_launch_shape {
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t block_x = 1;
  std::uint32_t block_y = 1;
  std::uint32_t block_z = 1;
  std::uint32_t dynamic_smem_bytes = 0;
  bool cooperative = false;
};

struct kernel_section {
  std::vector<kernel_symbol> symbols;
  cuda_launch_shape launch;
  bool stitched_persistent = false;
};

struct event_layout_entry {
  std::uint16_t event_slot = invalid_slot;
  std::uint16_t domain_rank = invalid_slot;
  std::uint32_t byte_offset = 0;
  megacu::memory_scope scope = megacu::memory_scope::local_device;
};

struct backend_section {
  std::uint16_t team_size = 1;
  std::uint32_t event_storage_bytes = 0;
  std::vector<event_layout_entry> events;
  bool requires_symmetric_partial_buffer = false;
  bool may_use_multimem_reduce = false;
};

struct target_metadata {
  std::string target_name;
  program_section program;
  dispatch_section dispatch;
  schedule_section schedule;
  kernel_section kernel;
  backend_section backend;
};

struct metadata_header {
  std::uint32_t magic = 0x4d435531;  // MCU1
  std::uint16_t version = 1;
  std::uint16_t header_bytes = sizeof(metadata_header);
  std::uint32_t total_bytes = sizeof(metadata_header);
  std::uint32_t checksum = 0;
  std::uint16_t extents = 0;
  std::uint16_t domains = 0;
  std::uint16_t participants = 0;
  std::uint16_t resources = 0;
  std::uint16_t events = 0;
  std::uint16_t submissions = 0;
  std::uint16_t team_size = 1;
  progress_model progress = progress_model::phased;
};

inline char const *to_string(progress_model progress) {
  switch (progress) {
    case progress_model::phased:
      return "phased";
    case progress_model::co_resident_persistent:
      return "co_resident_persistent";
  }
  return "unknown";
}

inline std::string to_json(target_metadata const &metadata) {
  std::string json;
  json += "{\"target\":\"";
  json += metadata.target_name;
  json += "\",\"progress\":\"";
  json += to_string(metadata.schedule.progress);
  json += "\",\"backend\":\"nvshmem\",\"team_size\":";
  json += std::to_string(metadata.backend.team_size);
  json += ",\"extents\":";
  json += std::to_string(metadata.program.extent_count);
  json += ",\"domains\":";
  json += std::to_string(metadata.program.domain_count);
  json += ",\"events\":";
  json += std::to_string(metadata.program.event_count);
  json += ",\"submissions\":";
  json += std::to_string(metadata.program.submission_count);
  json += "}";
  return json;
}

constexpr std::uint32_t mix(std::uint32_t result, std::uint32_t value) {
  return (result * 131u) + value;
}

constexpr std::uint32_t checksum(metadata_header header) {
  header.checksum = 0;
  std::uint32_t result = 0;
  result = mix(result, header.magic);
  result = mix(result, header.version);
  result = mix(result, header.header_bytes);
  result = mix(result, header.total_bytes);
  result = mix(result, header.extents);
  result = mix(result, header.domains);
  result = mix(result, header.participants);
  result = mix(result, header.resources);
  result = mix(result, header.events);
  result = mix(result, header.submissions);
  result = mix(result, header.team_size);
  result = mix(result, static_cast<std::uint8_t>(header.progress));
  return result;
}

constexpr metadata_header make_metadata_header(
    std::uint16_t extents,
    std::uint16_t domains,
    std::uint16_t participants,
    std::uint16_t resources,
    std::uint16_t events,
    std::uint16_t submissions,
    std::uint16_t team_size,
    progress_model progress) {
  metadata_header header;
  header.extents = extents;
  header.domains = domains;
  header.participants = participants;
  header.resources = resources;
  header.events = events;
  header.submissions = submissions;
  header.team_size = team_size;
  header.progress = progress;
  header.checksum = checksum(header);
  return header;
}

inline std::vector<std::byte> serialize(target_metadata const &metadata) {
  auto header = make_metadata_header(
      metadata.program.extent_count,
      metadata.program.domain_count,
      metadata.program.participant_count,
      metadata.program.resource_count,
      metadata.program.event_count,
      metadata.program.submission_count,
      metadata.backend.team_size,
      metadata.schedule.progress);

  std::vector<std::byte> bytes(sizeof(header));
  std::memcpy(bytes.data(), &header, sizeof(header));
  return bytes;
}

inline status validate(metadata_header header) {
  if (header.magic != 0x4d435531 || header.version != 1) {
    return {status_code::metadata_error, 2, "metadata header mismatch"};
  }
  if (header.header_bytes != sizeof(metadata_header) ||
      header.total_bytes != sizeof(metadata_header)) {
    return {status_code::metadata_error, 3, "metadata size mismatch"};
  }
  if (checksum(header) != header.checksum) {
    return {status_code::metadata_error, 4, "metadata checksum mismatch"};
  }

  if (header.extents == 0 || header.domains == 0 ||
      header.submissions == 0) {
    return {status_code::metadata_error, 5, "metadata missing required tables"};
  }

  return {};
}

inline status validate(std::span<const std::byte> bytes) {
  if (bytes.size() < sizeof(metadata_header)) {
    return {status_code::metadata_error, 1, "metadata too small"};
  }

  metadata_header header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  return validate(header);
}

}  // namespace megacu::detail
