#include "examples/cuda_nvshmem/gemm_allreduce/common/megacu_native_common.h"

extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  if (team.team_n_pes <= 1) {
    return gemm_ar_native::to_megacu(
        golden_phased_single_card_gemm_allreduce_f32(
            gemm_ar_native::make_workspace(workspace, events),
            gemm_ar_native::make_launch(launch),
            gemm_ar_native::make_problem(problem)));
  }

  return gemm_ar_native::to_megacu(
      golden_phased_multi_card_gemm_allreduce_f32(
          gemm_ar_native::make_workspace(workspace, events),
          gemm_ar_native::make_launch(launch),
          gemm_ar_native::make_team(team),
          gemm_ar_native::make_problem(problem)));
}
