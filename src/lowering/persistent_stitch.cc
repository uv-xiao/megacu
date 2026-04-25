#include <algorithm>
#include <cstdint>
#include <string>

#include <megacu/detail/components.h>

namespace megacu::detail {

kernel_section build_persistent_stitch_kernel_section(
    owned_program_ir const &program,
    schedule_section const &schedule) {
  kernel_section kernel;
  kernel.stitched_persistent =
      schedule.progress == progress_model::co_resident_persistent;
  kernel.launch = {
      .grid_x = schedule.residency.persistent_grid_blocks == 0
                    ? 1u
                    : schedule.residency.persistent_grid_blocks,
      .grid_y = 1,
      .grid_z = 1,
      .block_x = schedule.residency.threads_per_block == 0
                     ? 256u
                     : schedule.residency.threads_per_block,
      .block_y = 1,
      .block_z = 1,
      .dynamic_smem_bytes = 0,
      .cooperative = schedule.residency.requires_cooperative_launch};

  for (auto const &submission : program.submissions) {
    auto found = std::find_if(
        kernel.symbols.begin(),
        kernel.symbols.end(),
        [&](kernel_symbol const &symbol) {
          return symbol.op_slot == submission.op_slot ||
                 symbol.op_name == submission.op_name;
        });
    if (found != kernel.symbols.end()) {
      continue;
    }

    kernel.symbols.push_back({
        .op_name = std::string(submission.op_name),
        .op_slot = submission.op_slot,
        .role = entrypoint_role::callable_body,
        .symbol_id = static_cast<std::uint16_t>(kernel.symbols.size())});
  }

  return kernel;
}

}  // namespace megacu::detail

extern "C" int megacu_lowering_persistent_stitch_component() {
  return 1;
}
