#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cuda_runtime.h>
#include <nvshmem_host.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kSkipTest = 77;
constexpr int kM = 3;
constexpr int kN = 4;
constexpr int kK = 4;
constexpr int kLocalK = kK / 2;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBLocalBytes = kLocalK * kN * sizeof(float);
constexpr std::size_t kGatheredBBytes = kK * kN * sizeof(float);
constexpr std::size_t kCBytes = kM * kN * sizeof(float);

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

ag_gemm_problem correctness_problem() {
  return {.m = kM, .n = kN, .k = kK, .tile_m = 1, .tile_n = 2, .tile_k = 1};
}

void reset_outputs(void *out, void *gathered_b, void *events,
                   cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(gathered_b, 0, kGatheredBBytes, stream));
  require_cuda(cudaMemsetAsync(events, 0, 128, stream));
  require_cuda(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();
}

void check_expected(char const *label, int pe, void *out,
                    cudaStream_t stream) {
  std::vector<float> host_out(kM * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_out.data(), out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto got = host_out[row * kN + col];
      auto want = 6.0F;
      if (std::fabs(got - want) >= 1.0e-4F) {
        std::fprintf(stderr,
                     "%s pe=%d mismatch row=%d col=%d got=%.6f want=%.6f\n",
                     label, pe, row, col, got, want);
      }
      assert(std::fabs(got - want) < 1.0e-4F);
    }
  }
}

void run_case(char const *label,
              megacu::status (*fn)(ag_gemm_driver, float const *,
                                   float const *, float *, float *, void *,
                                   ag_gemm_problem),
              ag_gemm_driver driver, void const *a, void const *b,
              void *gathered_b, void *out, void *events, cudaStream_t stream,
              int pe) {
  reset_outputs(out, gathered_b, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b),
                   static_cast<float *>(gathered_b), static_cast<float *>(out),
                   events, correctness_problem());
  assert(status.code == megacu::status_code::ok);
  check_expected(label, pe, out, stream);
}

} // namespace

int main(int argc, char **argv) {
  char const *mode = argc > 1 ? argv[1] : "";
  nvshmem_init();

  int pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  if (npes != 2) {
    nvshmem_finalize();
    return kSkipTest;
  }

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    nvshmem_finalize();
    return kSkipTest;
  }

  int device = pe % device_count;
  require_cuda(cudaSetDevice(device));

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  void *a = nullptr;
  void *gathered_b = nullptr;
  void *out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&gathered_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));

  void *b = nvshmem_malloc(kBLocalBytes);
  void *events = nvshmem_malloc(128);
  assert(b != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, 1.0F);
  std::vector<float> host_b(kLocalK * kN, static_cast<float>(pe + 1));
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(b, host_b.data(), kBLocalBytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  ag_gemm_driver driver{
      .launch = {.stream = stream, .device_ordinal = device},
      .team = {.team = nullptr,
               .team_my_pe = pe,
               .team_n_pes = npes,
               .world_my_pe = pe,
               .world_n_pes = npes,
               .cuda_device_ordinal = device}};

  if (std::strcmp(mode, "megacu_host_orch") == 0) {
    run_case("ag-gemm host-orch", cuda_nvshmem_allgather_gemm_host_orch,
             driver, a, b, gathered_b, out, events, stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("ag-gemm seeded-orch", cuda_nvshmem_allgather_gemm_seeded_orch,
             driver, a, b, gathered_b, out, events, stream, pe);
  } else {
    assert(false && "unknown AG-GEMM two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(b);

  require_cuda(cudaFree(out));
  require_cuda(cudaFree(gathered_b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  nvshmem_finalize();
  return 0;
}
