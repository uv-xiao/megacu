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

struct dispatch_section {
  std::uint16_t work_entry_count = 0;
  std::uint16_t participant_entry_count = 0;
};

struct residency_group {
  std::uint16_t group = 0;
  std::uint16_t min_compute_workers = 0;
  std::uint16_t min_comm_workers = 0;
  bool all_workers_must_be_launched_together = false;
};

struct schedule_section {
  progress_model progress = progress_model::phased;
  std::vector<residency_group> residency_groups;
  bool has_blocking_device_wait = false;
};

struct kernel_symbol {
  std::string op_name;
  std::uint16_t op_slot = invalid_slot;
};

struct kernel_section {
  std::vector<kernel_symbol> symbols;
  bool stitched_persistent = false;
};

struct backend_section {
  std::uint16_t team_size = 1;
  std::uint32_t event_storage_bytes = 0;
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

inline std::uint32_t checksum(std::span<const std::byte> bytes) {
  std::uint32_t result = 0;
  for (auto byte : bytes) {
    result = (result * 131u) + static_cast<std::uint8_t>(byte);
  }
  return result;
}

inline std::vector<std::byte> serialize(target_metadata const &metadata) {
  metadata_header header;
  header.extents = metadata.program.extent_count;
  header.domains = metadata.program.domain_count;
  header.participants = metadata.program.participant_count;
  header.resources = metadata.program.resource_count;
  header.events = metadata.program.event_count;
  header.submissions = metadata.program.submission_count;
  header.team_size = metadata.backend.team_size;
  header.progress = metadata.schedule.progress;

  std::vector<std::byte> bytes(sizeof(header));
  std::memcpy(bytes.data(), &header, sizeof(header));
  auto *stored = reinterpret_cast<metadata_header *>(bytes.data());
  stored->checksum = 0;
  stored->checksum = checksum(bytes);
  return bytes;
}

inline status validate(std::span<const std::byte> bytes) {
  if (bytes.size() < sizeof(metadata_header)) {
    return {status_code::metadata_error, 1, "metadata too small"};
  }

  metadata_header header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.magic != 0x4d435531 || header.version != 1) {
    return {status_code::metadata_error, 2, "metadata header mismatch"};
  }
  if (header.total_bytes != bytes.size()) {
    return {status_code::metadata_error, 3, "metadata size mismatch"};
  }

  auto copy = std::vector<std::byte>(bytes.begin(), bytes.end());
  reinterpret_cast<metadata_header *>(copy.data())->checksum = 0;
  if (checksum(copy) != header.checksum) {
    return {status_code::metadata_error, 4, "metadata checksum mismatch"};
  }

  if (header.extents == 0 || header.domains == 0 ||
      header.submissions == 0) {
    return {status_code::metadata_error, 5, "metadata missing required tables"};
  }

  return {};
}

}  // namespace megacu::detail
