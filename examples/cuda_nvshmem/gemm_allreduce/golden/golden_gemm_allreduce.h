#pragma once

#include <cstdint>

enum class golden_status_code {
  ok,
  invalid_argument,
  launch_error,
  backend_error,
  unsupported,
};

struct golden_status {
  golden_status_code code = golden_status_code::ok;
  std::uint16_t detail = 0;
  char const *message = "";
};

struct golden_tensor_view {
  void *data = nullptr;
  std::int64_t bytes = 0;
};

struct golden_gemm_ar_workspace {
  golden_tensor_view a;
  golden_tensor_view b;
  golden_tensor_view partial;
  golden_tensor_view c;
  void *barriers = nullptr;
  std::int64_t barrier_bytes = 0;
};

struct golden_gemm_ar_problem {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int32_t tile_m = 0;
  std::int32_t tile_n = 0;
  std::int32_t compute_ctas = 0;
  std::int32_t comm_ctas = 0;
};

struct golden_gemm_ar_launch {
  void *stream = nullptr;
  int device_ordinal = 0;
};

struct golden_gemm_ar_team {
  void *team = nullptr;
  int team_my_pe = 0;
  int team_n_pes = 1;
  int world_my_pe = 0;
  int world_n_pes = 1;
  int cuda_device_ordinal = 0;
};

struct golden_gemm_ar_comm_ops {
  golden_status (*sum_reduce_f32)(
      void *team,
      void *dest,
      void const *src,
      std::int64_t elements,
      void *stream) = nullptr;
};

golden_status golden_phased_single_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem);

golden_status golden_overlap_single_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_problem problem);

golden_status golden_phased_multi_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem,
    golden_gemm_ar_comm_ops const *ops);

golden_status golden_overlap_multi_card_gemm_allreduce_f32(
    golden_gemm_ar_workspace workspace,
    golden_gemm_ar_launch launch,
    golden_gemm_ar_team team,
    golden_gemm_ar_problem problem,
    golden_gemm_ar_comm_ops const *ops);
