#include <array>
#include <cassert>
#include <cstddef>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

namespace {

gemm_ar_problem small_problem() {
  return {
      .m = 1,
      .n = 1,
      .k = 1,
      .tile_m = 1,
      .tile_n = 1};
}

}  // namespace

int main() {
  std::array<float, 1> a{};
  std::array<float, 1> b{};
  std::array<float, 1> c{};
  std::array<float, 2> partial{};
  std::array<std::byte, 128> events{};

  gemm_ar_driver driver{
      .launch = {.stream = nullptr, .device_ordinal = 0},
      .team = {
          .team = nullptr,
          .team_my_pe = 0,
          .team_n_pes = 2,
          .world_my_pe = 0,
          .world_n_pes = 2,
          .cuda_device_ordinal = 0}};

  auto phased = cuda_nvshmem_gemm_allreduce_phased_orchestrate(
      driver,
      a.data(),
      b.data(),
      partial.data(),
      c.data(),
      events.data(),
      small_problem());
  assert(phased.code == megacu::status_code::ok);

  return 0;
}
