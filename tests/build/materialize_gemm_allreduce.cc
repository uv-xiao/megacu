#include <cassert>
#include <string>

#include <megacu/detail/materialize.h>
#include <megacu/detail/target_metadata.h>

#include "examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce.h"

int main() {
  auto phased = megacu::detail::materialize_program<
      gemm_allreduce_phased_program>(
      megacu::detail::target_options{
          .target_name = "cuda_nvshmem_gemm_allreduce_phased",
          .progress = megacu::detail::progress_model::phased,
          .team_size = 2,
          .co_resident_guard = false});

  assert(phased.status.code == megacu::status_code::ok);
  assert(phased.program.extents.size() == 3);
  assert(phased.program.domains.size() == 2);
  assert(phased.program.participants.size() == 2);
  assert(phased.program.resources.size() == 2);
  assert(phased.program.events.size() == 1);
  assert(phased.program.submissions.size() == 2);
  assert(phased.metadata.schedule.progress ==
         megacu::detail::progress_model::phased);
  assert(phased.metadata.backend.team_size == 2);
  assert(phased.metadata.backend.requires_symmetric_partial_buffer);
  assert(phased.metadata.kernel.symbols.size() == 2);

  auto json = megacu::detail::to_json(phased.metadata);
  assert(json.find("\"target\":\"cuda_nvshmem_gemm_allreduce_phased\"") !=
         std::string::npos);
  assert(json.find("\"progress\":\"phased\"") != std::string::npos);
  assert(json.find("\"backend\":\"nvshmem\"") != std::string::npos);

  auto bytes = megacu::detail::serialize(phased.metadata);
  auto validation = megacu::detail::validate(bytes);
  assert(validation.code == megacu::status_code::ok);

  auto overlap = megacu::detail::materialize_program<
      gemm_allreduce_overlap_program>(
      megacu::detail::target_options{
          .target_name = "cuda_nvshmem_gemm_allreduce_overlap",
          .progress = megacu::detail::progress_model::co_resident_persistent,
          .team_size = 2,
          .co_resident_guard = true});

  assert(overlap.status.code == megacu::status_code::ok);
  assert(overlap.metadata.schedule.progress ==
         megacu::detail::progress_model::co_resident_persistent);
  assert(overlap.metadata.schedule.residency_groups.size() == 1);
  assert(overlap.metadata.schedule.residency_groups[0].min_compute_workers == 1);
  assert(overlap.metadata.schedule.residency_groups[0].min_comm_workers == 1);

  auto unsafe = megacu::detail::materialize_program<
      gemm_allreduce_overlap_program>(
      megacu::detail::target_options{
          .target_name = "unsafe_overlap",
          .progress = megacu::detail::progress_model::co_resident_persistent,
          .team_size = 2,
          .co_resident_guard = false});

  assert(unsafe.status.code == megacu::status_code::unsupported);

  return 0;
}
