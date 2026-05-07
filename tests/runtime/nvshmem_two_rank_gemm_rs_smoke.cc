#include "examples/cuda_nvshmem/gemm_reduce_scatter/common/gemm_reduce_scatter.h"

#include <cuda_runtime.h>
#include <nvshmem_host.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kSkipTest = 77;
constexpr int kM = 4;
constexpr int kN = 3;
constexpr int kK = 4;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBBytes = kK * kN * sizeof(float);
constexpr std::size_t kCBytes = kM * kN * sizeof(float);

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

gemm_rs_problem correctness_problem() {
  return {.m = kM, .n = kN, .k = kK, .tile_m = 1, .tile_n = 2};
}

void reset_outputs(void *out, void *partial, void *events,
                   cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(partial, 0, kCBytes, stream));
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
      auto want = row % 2 == pe ? 12.0F : 0.0F;
      auto got = host_out[row * kN + col];
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
              megacu::status (*fn)(gemm_rs_driver, float const *,
                                   float const *, float *, float *, void *,
                                   gemm_rs_problem),
              gemm_rs_driver driver, void const *a, void const *b,
              void *partial, void *out, void *events, cudaStream_t stream,
              int pe) {
  reset_outputs(out, partial, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b), static_cast<float *>(partial),
                   static_cast<float *>(out), events, correctness_problem());
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
  void *b = nullptr;
  void *out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));

  void *partial = nvshmem_malloc(kCBytes);
  void *events = nvshmem_malloc(128);
  assert(partial != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, static_cast<float>(pe + 1));
  std::vector<float> host_b(kK * kN, 1.0F);
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(b, host_b.data(), kBBytes,
                               cudaMemcpyHostToDevice, stream));

  gemm_rs_driver driver{
      .launch = {.stream = stream, .device_ordinal = device},
      .team = {.team = nullptr,
               .team_my_pe = pe,
               .team_n_pes = npes,
               .world_my_pe = pe,
               .world_n_pes = npes,
               .cuda_device_ordinal = device}};

  if (std::strcmp(mode, "megacu_host_orch") == 0) {
    run_case("gemm-rs host-orch", cuda_nvshmem_gemm_reduce_scatter_host_orch,
             driver, a, b, partial, out, events, stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("gemm-rs seeded-orch",
             cuda_nvshmem_gemm_reduce_scatter_seeded_orch, driver, a, b,
             partial, out, events, stream, pe);
  } else {
    assert(false && "unknown GEMM-RS two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(partial);

  require_cuda(cudaFree(out));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  nvshmem_finalize();
  return 0;
}
