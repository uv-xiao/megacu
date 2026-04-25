#pragma once

#include <megacu/detail/materialize.h>
#include <megacu/detail/target_metadata.h>

namespace megacu::detail {

dispatch_section build_tiled_compute_comm_dispatch(
    owned_program_ir const &program,
    target_options options);

schedule_section build_static_persistent_schedule(
    owned_program_ir const &program,
    dispatch_section const &dispatch,
    target_options options);

kernel_section build_persistent_stitch_kernel_section(
    owned_program_ir const &program,
    schedule_section const &schedule);

backend_section build_nvshmem_backend_section(
    owned_program_ir const &program,
    target_options options);

}  // namespace megacu::detail
