#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kM = 3;
constexpr int kN = 4;
constexpr int kK = 3;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBBytes = kK * kN * sizeof(float);
constexpr std::size_t kCBytes = kM * kN * sizeof(float);

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

int choose_device() {
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count));
  assert(count > 0);

  if (char const *env = std::getenv("MEGACU_TEST_CUDA_DEVICE")) {
    char *end = nullptr;
    long requested = std::strtol(env, &end, 10);
    assert(end != env && *end == '\0');
    assert(requested >= 0 && requested < count);
    return static_cast<int>(requested);
  }

  return count - 1;
}

float expected(std::vector<float> const &a, std::vector<float> const &b,
               int row, int col) {
  float value = 0.0F;
  for (int kk = 0; kk < kK; ++kk) {
    value += a[row * kK + kk] * b[kk * kN + col];
  }
  return value;
}

void check_expected(std::vector<float> const &host_a,
                    std::vector<float> const &host_b,
                    std::vector<float> const &host_c, char const *label) {
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      float want = expected(host_a, host_b, row, col);
      float got = host_c[row * kN + col];
      if (std::fabs(got - want) >= 1.0e-4F) {
        std::cerr << label << " mismatch row=" << row << " col=" << col
                  << " got=" << got << " want=" << want << "\n";
      }
      assert(std::fabs(got - want) < 1.0e-4F);
    }
  }
}

void reset_outputs(void *out, void *gathered_b, void *events,
                   cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(gathered_b, 0, kBBytes, stream));
  if (events != nullptr) {
    require_cuda(cudaMemsetAsync(events, 0, 128, stream));
  }
}

void run_case(char const *label,
              megacu::status (*fn)(ag_gemm_driver, float const *,
                                   float const *, float *, float *,
                                   ag_gemm_problem),
              ag_gemm_driver driver, void const *a, void const *b,
              void *gathered_b, void *out, cudaStream_t stream,
              std::vector<float> const &host_a,
              std::vector<float> const &host_b,
              std::vector<float> &host_c) {
  reset_outputs(out, gathered_b, nullptr, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b),
                   static_cast<float *>(gathered_b), static_cast<float *>(out),
                   ag_gemm_problem{
                       .m = kM,
                       .n = kN,
                       .k = kK,
                       .tile_m = 1,
                       .tile_n = 2,
                       .tile_k = 1});
  assert(status.code == megacu::status_code::ok);
  require_cuda(cudaMemcpyAsync(host_c.data(), out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  check_expected(host_a, host_b, host_c, label);
}

void run_megacu_case(char const *label,
                     megacu::status (*fn)(ag_gemm_driver, float const *,
                                          float const *, float *, float *,
                                          void *, ag_gemm_problem),
                     ag_gemm_driver driver, void const *a, void const *b,
                     void *gathered_b, void *out, void *events,
                     cudaStream_t stream, std::vector<float> const &host_a,
                     std::vector<float> const &host_b,
                     std::vector<float> &host_c) {
  reset_outputs(out, gathered_b, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b),
                   static_cast<float *>(gathered_b), static_cast<float *>(out),
                   events,
                   ag_gemm_problem{
                       .m = kM,
                       .n = kN,
                       .k = kK,
                       .tile_m = 1,
                       .tile_n = 2,
                       .tile_k = 1});
  assert(status.code == megacu::status_code::ok);
  require_cuda(cudaMemcpyAsync(host_c.data(), out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  check_expected(host_a, host_b, host_c, label);
}

} // namespace

int main() {
  int device = choose_device();
  require_cuda(cudaSetDevice(device));

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  std::vector<float> host_a{1.0F, 2.0F, 3.0F, 2.0F, 1.0F,
                            0.5F, 0.5F, 1.5F, 2.5F};
  std::vector<float> host_b{1.0F, 0.0F, 2.0F, 1.0F, 0.5F, 1.0F,
                            0.0F, 2.0F, 2.0F, 1.5F, 1.0F, 0.5F};
  std::vector<float> host_c(kM * kN, 0.0F);

  void *a = nullptr;
  void *b = nullptr;
  void *gathered_b = nullptr;
  void *out = nullptr;
  void *events = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&gathered_b, kBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));
  require_cuda(cudaMalloc(&events, 128));

  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(b, host_b.data(), kBBytes,
                               cudaMemcpyHostToDevice, stream));

  ag_gemm_driver driver{
      .launch = {.stream = stream, .device_ordinal = device},
      .team = {.team = nullptr,
               .team_my_pe = 0,
               .team_n_pes = 1,
               .world_my_pe = 0,
               .world_n_pes = 1,
               .cuda_device_ordinal = device}};

  run_case("golden_single", golden_cuda_nvshmem_allgather_gemm, driver, a, b,
           gathered_b, out, stream, host_a, host_b, host_c);
  run_case("baseline_single", baseline_cuda_nvshmem_allgather_gemm, driver, a,
           b, gathered_b, out, stream, host_a, host_b, host_c);
  run_megacu_case("megacu_host_orch_single",
                  cuda_nvshmem_allgather_gemm_host_orch, driver, a, b,
                  gathered_b, out, events, stream, host_a, host_b, host_c);
  run_megacu_case("megacu_seeded_orch_single",
                  cuda_nvshmem_allgather_gemm_seeded_orch, driver, a, b,
                  gathered_b, out, events, stream, host_a, host_b, host_c);

  require_cuda(cudaFree(events));
  require_cuda(cudaFree(out));
  require_cuda(cudaFree(gathered_b));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));
  return 0;
}
