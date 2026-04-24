#include <cassert>

#include <megacu/detail/target_metadata.h>

extern "C" megacu::detail::metadata_header const
    megacu_cuda_nvshmem_gemm_allreduce_phased_metadata;
extern "C" megacu::detail::metadata_header const
    megacu_cuda_nvshmem_gemm_allreduce_overlap_metadata;

int main() {
  auto phased = megacu_cuda_nvshmem_gemm_allreduce_phased_metadata;
  auto overlap = megacu_cuda_nvshmem_gemm_allreduce_overlap_metadata;

  assert(megacu::detail::validate(phased).code == megacu::status_code::ok);
  assert(megacu::detail::validate(overlap).code == megacu::status_code::ok);
  assert(phased.progress == megacu::detail::progress_model::phased);
  assert(overlap.progress ==
         megacu::detail::progress_model::co_resident_persistent);
  assert(phased.extents == 3);
  assert(phased.domains == 2);
  assert(phased.team_size == 2);
  assert(overlap.team_size == 2);

  return 0;
}
