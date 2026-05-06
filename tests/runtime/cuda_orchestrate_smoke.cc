#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

namespace {

constexpr std::size_t kBytes = 256;

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

gemm_ar_problem small_problem() {
  return {
      .m = 2,
      .n = 2,
      .k = 2,
      .tile_m = 1,
      .tile_n = 1};
}

}  // namespace

int main() {
  int device = choose_device();
  require_cuda(cudaSetDevice(device));

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  void *partial = nullptr;
  void *events = nullptr;
  void *marker = nullptr;

  require_cuda(cudaMalloc(&a, kBytes));
  require_cuda(cudaMalloc(&b, kBytes));
  require_cuda(cudaMalloc(&c, kBytes));
  require_cuda(cudaMalloc(&partial, kBytes));
  require_cuda(cudaMalloc(&events, kBytes));
  require_cuda(cudaMalloc(&marker, sizeof(std::uint32_t)));

  require_cuda(cudaMemsetAsync(marker, 0x5a, sizeof(std::uint32_t), stream));
  std::uint32_t host_marker = 0;
  require_cuda(cudaMemcpyAsync(
      &host_marker,
      marker,
      sizeof(host_marker),
      cudaMemcpyDeviceToHost,
      stream));
  require_cuda(cudaStreamSynchronize(stream));
  assert(host_marker == 0x5a5a5a5a);

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
      small_problem());
  assert(status.code == megacu::status_code::ok);

  require_cuda(cudaFree(marker));
  require_cuda(cudaFree(events));
  require_cuda(cudaFree(partial));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));
  return 0;
}
