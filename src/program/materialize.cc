#include <cstdint>
#include <string>

#include <megacu/detail/components.h>
#include <megacu/detail/materialize.h>

namespace megacu::detail {

target_metadata lower_target(
    owned_program_ir const &program,
    target_options options) {
  target_metadata metadata;
  metadata.target_name = std::string(options.target_name);
  metadata.program.extent_count =
      static_cast<std::uint16_t>(program.extents.size());
  metadata.program.domain_count =
      static_cast<std::uint16_t>(program.domains.size());
  metadata.program.participant_count =
      static_cast<std::uint16_t>(program.participants.size());
  metadata.program.resource_count =
      static_cast<std::uint16_t>(program.resources.size());
  metadata.program.event_count =
      static_cast<std::uint16_t>(program.events.size());
  metadata.program.submission_count =
      static_cast<std::uint16_t>(program.submissions.size());

  metadata.dispatch = build_tiled_compute_comm_dispatch(program, options);
  metadata.schedule =
      build_static_persistent_schedule(program, metadata.dispatch, options);
  metadata.kernel =
      build_persistent_stitch_kernel_section(program, metadata.schedule);
  metadata.backend = build_nvshmem_backend_section(program, options);

  return metadata;
}

}  // namespace megacu::detail
