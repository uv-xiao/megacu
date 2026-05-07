#pragma once

#include <cstdint>

#include <megacu/runtime.h>

using ag_gemm_driver = megacu::cuda_nvshmem::driver_view;

struct ag_gemm_problem {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int32_t tile_m = 0;
  std::int32_t tile_n = 0;
  std::int32_t tile_k = 0;
};

megacu::status golden_cuda_nvshmem_allgather_gemm(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, ag_gemm_problem problem);

megacu::status baseline_cuda_nvshmem_allgather_gemm(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, ag_gemm_problem problem);

megacu::status cuda_nvshmem_allgather_gemm_host_orch(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, void *events, ag_gemm_problem problem);

megacu::status cuda_nvshmem_allgather_gemm_seeded_orch(
    ag_gemm_driver driver, float const *a, float const *b, float *gathered_b,
    float *out, void *events, ag_gemm_problem problem);
