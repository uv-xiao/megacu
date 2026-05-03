#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h"

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_driver driver,
    float const *a,
    float const *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem) {
  return gemm_ar_detail::orchestrate_impl(
      megacu::runtime::progress_model::asap,
      driver,
      a,
      b,
      partial,
      out,
      events,
      problem);
}
