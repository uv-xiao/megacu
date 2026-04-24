#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"

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
      .m = 16,
      .n = 16,
      .k = 16,
      .tile_m = 16,
      .tile_n = 16};
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
  void *scratch = nullptr;
  void *marker = nullptr;

  require_cuda(cudaMalloc(&a, kBytes));
  require_cuda(cudaMalloc(&b, kBytes));
  require_cuda(cudaMalloc(&c, kBytes));
  require_cuda(cudaMalloc(&partial, kBytes));
  require_cuda(cudaMalloc(&events, kBytes));
  require_cuda(cudaMalloc(&scratch, kBytes));
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

  megacu::backend_id backend{1};
  megacu::session_id session{7};

  gemm_ar_workspace workspace{
      .a = {.data = a, .bytes = static_cast<std::int64_t>(kBytes)},
      .b = {.data = b, .bytes = static_cast<std::int64_t>(kBytes)},
      .partial = {
          .buffer = {
              .data = partial,
              .bytes = static_cast<std::int64_t>(kBytes),
              .backend = backend,
              .session = session},
          .type = megacu::dtype::f32},
      .c = {.data = c, .bytes = static_cast<std::int64_t>(kBytes)},
      .scratch = megacu::span<std::byte>{
          static_cast<std::byte *>(scratch),
          kBytes}};

  megacu::event_storage_view event_storage{
      .buffer = {
          .data = events,
          .bytes = static_cast<std::int64_t>(kBytes),
          .backend = backend,
          .session = session}};

  megacu::cuda::launch_view launch{
      .stream = stream,
      .device_ordinal = device};
  megacu::nvshmem::team_view team{
      .team = nullptr,
      .team_my_pe = 0,
      .team_n_pes = 2,
      .world_my_pe = 0,
      .world_n_pes = 2,
      .cuda_device_ordinal = device,
      .backend = backend,
      .session = session};

  auto phased = cuda_nvshmem_gemm_allreduce_phased_orchestrate(
      workspace, event_storage, launch, team, small_problem());
  assert(phased.code == megacu::status_code::ok);

  auto overlap = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, event_storage, launch, team, small_problem());
  assert(overlap.code == megacu::status_code::ok);

  require_cuda(cudaFree(marker));
  require_cuda(cudaFree(scratch));
  require_cuda(cudaFree(events));
  require_cuda(cudaFree(partial));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));
  return 0;
}
