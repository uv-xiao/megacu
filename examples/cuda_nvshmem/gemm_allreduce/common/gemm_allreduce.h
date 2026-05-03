#pragma once

#include <cstdint>

#include <megacu/runtime.h>
#include <megacu/views.h>

namespace ops {

struct gemm_tile_produce {
  static constexpr auto name = "gemm_tile_produce";
};

struct allreduce_tile_consume {
  static constexpr auto name = "allreduce_tile_consume";
};

struct phased_megakernel {
  static constexpr auto name = "phased_megakernel";
};

}  // namespace ops

using gemm_ar_driver = megacu::cuda_nvshmem::driver_view;

struct gemm_ar_problem {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int32_t tile_m = 0;
  std::int32_t tile_n = 0;
};

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_driver driver,
    float const *a,
    float const *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem);
