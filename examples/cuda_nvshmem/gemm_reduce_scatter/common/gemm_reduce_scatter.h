#pragma once

#include <cstdint>

#include <megacu/runtime.h>

using gemm_rs_driver = megacu::cuda_nvshmem::driver_view;

struct gemm_rs_problem {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int32_t tile_m = 0;
  std::int32_t tile_n = 0;
};

megacu::status golden_cuda_nvshmem_gemm_reduce_scatter(
    gemm_rs_driver driver, float const *a, float const *b, float *partial,
    float *out, gemm_rs_problem problem);

megacu::status baseline_cuda_nvshmem_gemm_reduce_scatter(
    gemm_rs_driver driver, float const *a, float const *b, float *partial,
    float *out, gemm_rs_problem problem);

megacu::status cuda_nvshmem_gemm_reduce_scatter_host_orch(
    gemm_rs_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_rs_problem problem);

megacu::status cuda_nvshmem_gemm_reduce_scatter_seeded_orch(
    gemm_rs_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_rs_problem problem);
