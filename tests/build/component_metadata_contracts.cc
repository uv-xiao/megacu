#include <cassert>
#include <string>

#include <megacu/detail/materialize.h>
#include <megacu/detail/target_metadata.h>

#include "examples/cuda_nvshmem/common/gemm_allreduce/gemm_allreduce.h"

namespace {

using megacu::detail::entrypoint_role;
using megacu::detail::event_wait_mode;
using megacu::detail::placement_role;
using megacu::detail::progress_model;
using megacu::detail::schedule_action;

void check_phased() {
  auto target = megacu::detail::materialize_program<
      gemm_allreduce_phased_program>(
      megacu::detail::target_options{
          .target_name = "cuda_nvshmem_gemm_allreduce_phased",
          .progress = progress_model::phased,
          .team_size = 2,
          .co_resident_guard = false});

  assert(target.status.code == megacu::status_code::ok);

  assert(target.metadata.dispatch.work.size() == 2);
  assert(target.metadata.dispatch.work[0].role == placement_role::compute);
  assert(target.metadata.dispatch.work[0].participant_slot == 0);
  assert(target.metadata.dispatch.work[1].role == placement_role::comm);
  assert(target.metadata.dispatch.work[1].participant_slot == 1);

  assert(target.metadata.dispatch.participants.size() == 4);
  assert(target.metadata.dispatch.participants[0].backend_peer == 0);
  assert(target.metadata.dispatch.participants[1].backend_peer == 1);

  assert(target.metadata.schedule.entries.size() == 2);
  assert(target.metadata.schedule.entries[0].action ==
         schedule_action::launch_gemm_tile_produce);
  assert(target.metadata.schedule.entries[0].released_event_slot == 0);
  assert(target.metadata.schedule.entries[1].action ==
         schedule_action::launch_allreduce_tile_consume);
  assert(target.metadata.schedule.entries[1].required_event_slot == 0);
  assert(target.metadata.schedule.entries[1].wait_mode ==
         event_wait_mode::phase_satisfied);
  assert(target.metadata.schedule.residency_groups.empty());

  assert(target.metadata.kernel.symbols.size() == 2);
  assert(target.metadata.kernel.symbols[0].role == entrypoint_role::callable_body);
  assert(target.metadata.kernel.symbols[1].role == entrypoint_role::callable_body);
  assert(!target.metadata.kernel.stitched_persistent);

  assert(target.metadata.backend.events.size() == 1);
  assert(target.metadata.backend.events[0].event_slot == 0);
  assert(target.metadata.backend.events[0].domain_rank == 1);
  assert(target.metadata.backend.events[0].byte_offset == 0);
}

void check_overlap() {
  auto target = megacu::detail::materialize_program<
      gemm_allreduce_overlap_program>(
      megacu::detail::target_options{
          .target_name = "cuda_nvshmem_gemm_allreduce_overlap",
          .progress = progress_model::co_resident_persistent,
          .team_size = 2,
          .co_resident_guard = true});

  assert(target.status.code == megacu::status_code::ok);
  assert(target.metadata.schedule.entries.size() == 2);
  assert(target.metadata.schedule.entries[1].wait_mode ==
         event_wait_mode::blocking_device_wait);
  assert(target.metadata.schedule.residency_groups.size() == 1);
  assert(target.metadata.schedule.residency_groups[0]
             .all_workers_must_be_launched_together);
  assert(target.metadata.schedule.residency.persistent_grid_blocks == 2);
  assert(target.metadata.schedule.residency.threads_per_block > 0);
  assert(target.metadata.kernel.stitched_persistent);
}

}  // namespace

extern "C" int megacu_dispatcher_tiled_compute_comm_component();
extern "C" int megacu_scheduler_static_persistent_component();
extern "C" int megacu_lowering_persistent_stitch_component();
extern "C" int megacu_platform_cuda_component();
extern "C" int megacu_backend_nvshmem_component();

int main() {
  check_phased();
  check_overlap();

  assert(megacu_dispatcher_tiled_compute_comm_component() == 1);
  assert(megacu_scheduler_static_persistent_component() == 1);
  assert(megacu_lowering_persistent_stitch_component() == 1);
  assert(megacu_platform_cuda_component() == 1);
  assert(megacu_backend_nvshmem_component() == 1);

  return 0;
}
