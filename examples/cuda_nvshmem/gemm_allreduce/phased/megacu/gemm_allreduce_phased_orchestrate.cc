#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h"

extern "C" megacu::detail::metadata_header const
    megacu_cuda_nvshmem_gemm_allreduce_phased_metadata =
        megacu::detail::make_metadata_header(
            3,
            2,
            2,
            2,
            1,
            2,
            2,
            megacu::detail::progress_model::phased);

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  return gemm_ar_detail::orchestrate_impl(
      workspace,
      events,
      launch,
      team,
      problem,
      megacu_cuda_nvshmem_gemm_allreduce_phased_metadata);
}
