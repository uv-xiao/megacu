#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"
#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

namespace {

constexpr int kM = 2;
constexpr int kN = 3;
constexpr int kK = 4;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBBytes = kK * kN * sizeof(float);
constexpr std::size_t kCBytes = kM * kN * sizeof(float);
constexpr std::size_t kPartialBytes = 2 * kCBytes;

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

float expected(std::vector<float> const &a,
               std::vector<float> const &b,
               int row,
               int col) {
  float value = 0.0f;
  for (int kk = 0; kk < kK; ++kk) {
    value += a[row * kK + kk] * b[kk * kN + col];
  }
  return value;
}

void check_expected(std::vector<float> const &host_a,
                    std::vector<float> const &host_b,
                    std::vector<float> const &host_c,
                    float allreduce_scale,
                    char const *label) {
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      float want = allreduce_scale * expected(host_a, host_b, row, col);
      float got = host_c[row * kN + col];
      if (std::fabs(got - want) >= 1.0e-4f) {
        std::cerr << label << " mismatch row=" << row << " col=" << col
                  << " got=" << got << " want=" << want << "\n";
      }
      assert(std::fabs(got - want) < 1.0e-4f);
    }
  }
}

void reset_outputs(
    void *c,
    void *partial,
    void *events,
    cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(c, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(partial, 0, kPartialBytes, stream));
  require_cuda(cudaMemsetAsync(events, 0, 128, stream));
}

}  // namespace

int main() {
  int device = choose_device();
  require_cuda(cudaSetDevice(device));

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  std::vector<float> host_a{
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.5f, 3.0f};
  std::vector<float> host_b{
      1.0f, 0.0f, 2.0f,
      0.5f, 1.0f, 0.0f,
      2.0f, 1.5f, 1.0f,
      1.0f, 2.0f, 0.5f};
  std::vector<float> host_c(kM * kN, 0.0f);

  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  void *partial = nullptr;
  void *events = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&c, kCBytes));
  require_cuda(cudaMalloc(&partial, kPartialBytes));
  require_cuda(cudaMalloc(&events, 128));

  require_cuda(cudaMemcpyAsync(
      a, host_a.data(), kABytes, cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(
      b, host_b.data(), kBBytes, cudaMemcpyHostToDevice, stream));
  reset_outputs(c, partial, events, stream);

  golden_gemm_ar_workspace golden_workspace{
      .a = {.data = a, .bytes = static_cast<std::int64_t>(kABytes)},
      .b = {.data = b, .bytes = static_cast<std::int64_t>(kBBytes)},
      .partial = {
          .data = partial,
          .bytes = static_cast<std::int64_t>(kPartialBytes)},
      .c = {.data = c, .bytes = static_cast<std::int64_t>(kCBytes)},
      .barriers = events,
      .barrier_bytes = 128};
  golden_gemm_ar_launch golden_launch{
      .stream = stream,
      .device_ordinal = device};
  golden_gemm_ar_problem golden_problem{
      .m = kM,
      .n = kN,
      .k = kK,
      .tile_m = 1,
      .tile_n = 2,
      .compute_ctas = 2,
      .comm_ctas = 1};

  auto golden_status = golden_phased_single_card_gemm_allreduce_f32(
      golden_workspace, golden_launch, golden_problem);
  assert(golden_status.code == golden_status_code::ok);
  require_cuda(cudaMemcpyAsync(
      host_c.data(), c, kCBytes, cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  check_expected(host_a, host_b, host_c, 1.0f, "golden_phased_single");

  reset_outputs(c, partial, events, stream);

  gemm_ar_driver driver{
      .launch = {.stream = stream, .device_ordinal = device},
      .team = {
          .team = nullptr,
          .team_my_pe = 0,
          .team_n_pes = 1,
          .world_my_pe = 0,
          .world_n_pes = 1,
          .cuda_device_ordinal = device}};

  auto status = cuda_nvshmem_gemm_allreduce_host_orch(
      driver,
      static_cast<float const *>(a),
      static_cast<float const *>(b),
      static_cast<float *>(partial),
      static_cast<float *>(c),
      events,
      gemm_ar_problem{.m = kM, .n = kN, .k = kK, .tile_m = 1, .tile_n = 2});
  assert(status.code == megacu::status_code::ok);

  require_cuda(cudaMemcpyAsync(
      host_c.data(), c, kCBytes, cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  check_expected(host_a, host_b, host_c, 1.0f, "megacu_host_orch_single");

  require_cuda(cudaFree(events));
  require_cuda(cudaFree(partial));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));
  return 0;
}
