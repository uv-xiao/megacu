#include <cstdint>

#include <megacu/detail/components.h>

namespace megacu::detail {

backend_section build_nvshmem_backend_section(
    owned_program_ir const &program,
    target_options options) {
  backend_section backend;
  backend.team_size = options.team_size;
  backend.event_storage_bytes =
      static_cast<std::uint32_t>(program.events.size() * options.team_size *
                                 sizeof(std::uint64_t));
  backend.requires_symmetric_partial_buffer = true;
  backend.may_use_multimem_reduce = true;

  for (auto const &event : program.events) {
    backend.events.push_back({
        .event_slot = event.slot,
        .domain_rank = event.domain_slots.empty()
            ? invalid_slot
            : static_cast<std::uint16_t>(event.domain_slots.size() - 1),
        .byte_offset = static_cast<std::uint32_t>(
            event.slot * options.team_size * sizeof(std::uint64_t)),
        .scope = event.scope});
  }

  return backend;
}

}  // namespace megacu::detail

extern "C" int megacu_backend_nvshmem_component() {
  return 1;
}
