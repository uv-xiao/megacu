#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

#include <cuda_runtime.h>
#include <nvshmem_host.h>

#include <cassert>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kSkipTest = 77;

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

void run_mode(char const *label,
              int (*fn)(megacu::examples::tiny_decode::buffers *), int pe) {
  megacu::examples::tiny_decode::buffers golden{};
  megacu::examples::tiny_decode::buffers candidate{};
  megacu::examples::tiny_decode::seed_inputs(golden);
  candidate = golden;

  auto golden_status = megacu_tiny_decode_golden(&golden);
  auto candidate_status = fn(&candidate);
  if (golden_status != 0 || candidate_status != 0 ||
      !megacu::examples::tiny_decode::nearly_equal(golden, candidate)) {
    std::fprintf(stderr,
                 "%s pe=%d failed golden_status=%d candidate_status=%d\n",
                 label, pe, golden_status, candidate_status);
  }
  assert(golden_status == 0);
  assert(candidate_status == 0);
  assert(megacu::examples::tiny_decode::nearly_equal(golden, candidate));
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
  nvshmem_barrier_all();

  if (std::strcmp(mode, "megacu_host_orch") == 0) {
    run_mode("tiny-decode host-orch", megacu_tiny_decode_megacu_host_orch, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_mode("tiny-decode seeded-orch", megacu_tiny_decode_megacu_seeded_orch,
             pe);
  } else {
    assert(false && "unknown tiny decode two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_finalize();
  return 0;
}
