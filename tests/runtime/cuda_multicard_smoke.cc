#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "examples/cuda_nvshmem/common/gemm_allreduce/gemm_allreduce.h"

namespace {

constexpr int kSkipTest = 77;
constexpr std::size_t kBytes = 256;

struct selected_devices {
  int first = 0;
  int second = 1;
};

struct device_allocation {
  int device = 0;
  cudaStream_t stream = nullptr;
  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  void *partial = nullptr;
  void *events = nullptr;
  void *scratch = nullptr;
  void *marker = nullptr;
};

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

selected_devices choose_devices(int count) {
  assert(count >= 2);

  if (char const *env = std::getenv("MEGACU_TEST_CUDA_DEVICES")) {
    char *end = nullptr;
    long first = std::strtol(env, &end, 10);
    assert(end != env && *end == ',');
    char *second_begin = end + 1;
    long second = std::strtol(second_begin, &end, 10);
    assert(end != second_begin && *end == '\0');
    assert(first >= 0 && first < count);
    assert(second >= 0 && second < count);
    assert(first != second);
    return {static_cast<int>(first), static_cast<int>(second)};
  }

  return {.first = count - 2, .second = count - 1};
}

gemm_ar_problem small_problem() {
  return {
      .m = 2,
      .n = 2,
      .k = 2,
      .tile_m = 1,
      .tile_n = 1};
}

device_allocation allocate_device(int device) {
  device_allocation allocation{.device = device};
  require_cuda(cudaSetDevice(device));
  require_cuda(cudaStreamCreate(&allocation.stream));
  require_cuda(cudaMalloc(&allocation.a, kBytes));
  require_cuda(cudaMalloc(&allocation.b, kBytes));
  require_cuda(cudaMalloc(&allocation.c, kBytes));
  require_cuda(cudaMalloc(&allocation.partial, kBytes));
  require_cuda(cudaMalloc(&allocation.events, kBytes));
  require_cuda(cudaMalloc(&allocation.scratch, kBytes));
  require_cuda(cudaMalloc(&allocation.marker, sizeof(std::uint32_t)));
  return allocation;
}

void free_device(device_allocation allocation) {
  require_cuda(cudaSetDevice(allocation.device));
  require_cuda(cudaFree(allocation.marker));
  require_cuda(cudaFree(allocation.scratch));
  require_cuda(cudaFree(allocation.events));
  require_cuda(cudaFree(allocation.partial));
  require_cuda(cudaFree(allocation.c));
  require_cuda(cudaFree(allocation.b));
  require_cuda(cudaFree(allocation.a));
  require_cuda(cudaStreamDestroy(allocation.stream));
}

void enable_peer_access_if_available(int from, int to) {
  int can_access_peer = 0;
  require_cuda(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
  if (can_access_peer == 0) {
    return;
  }

  require_cuda(cudaSetDevice(from));
  cudaError_t status = cudaDeviceEnablePeerAccess(to, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
    return;
  }
  require_cuda(status);
}

gemm_ar_workspace workspace_for(device_allocation allocation,
                                megacu::backend_id backend,
                                megacu::session_id session) {
  return {
      .a = {
          .data = allocation.a,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
      .b = {
          .data = allocation.b,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
      .partial = {
          .buffer = {
              .data = allocation.partial,
              .bytes = static_cast<std::int64_t>(kBytes),
              .backend = backend,
              .session = session},
          .type = megacu::dtype::f32},
      .c = {
          .data = allocation.c,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
      .scratch = megacu::span<std::byte>{
          static_cast<std::byte *>(allocation.scratch),
          kBytes}};
}

megacu::event_storage_view events_for(device_allocation allocation,
                                      megacu::backend_id backend,
                                      megacu::session_id session) {
  return {
      .buffer = {
          .data = allocation.events,
          .bytes = static_cast<std::int64_t>(kBytes),
          .backend = backend,
          .session = session}};
}

megacu::status orchestrate_on(device_allocation allocation,
                              int logical_pe,
                              megacu::backend_id backend,
                              megacu::session_id session) {
  megacu::cuda::launch_view launch{
      .stream = allocation.stream,
      .device_ordinal = allocation.device};
  megacu::nvshmem::team_view team{
      .team = nullptr,
      .team_my_pe = 0,
      .team_n_pes = 1,
      .world_my_pe = logical_pe,
      .world_n_pes = 2,
      .cuda_device_ordinal = allocation.device,
      .backend = backend,
      .session = session};

  return cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace_for(allocation, backend, session),
      events_for(allocation, backend, session),
      launch,
      team,
      small_problem());
}

}  // namespace

int main() {
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count));
  if (count < 2) {
    return kSkipTest;
  }

  selected_devices devices = choose_devices(count);
  enable_peer_access_if_available(devices.first, devices.second);
  enable_peer_access_if_available(devices.second, devices.first);

  device_allocation first = allocate_device(devices.first);
  device_allocation second = allocate_device(devices.second);

  require_cuda(cudaSetDevice(first.device));
  require_cuda(cudaMemsetAsync(
      first.marker, 0x31, sizeof(std::uint32_t), first.stream));
  require_cuda(cudaStreamSynchronize(first.stream));

  require_cuda(cudaSetDevice(second.device));
  require_cuda(cudaMemcpyPeerAsync(
      second.marker,
      second.device,
      first.marker,
      first.device,
      sizeof(std::uint32_t),
      second.stream));
  std::uint32_t host_marker = 0;
  require_cuda(cudaMemcpyAsync(
      &host_marker,
      second.marker,
      sizeof(host_marker),
      cudaMemcpyDeviceToHost,
      second.stream));
  require_cuda(cudaStreamSynchronize(second.stream));
  assert(host_marker == 0x31313131);

  megacu::backend_id backend{1};
  megacu::session_id session{7};
  assert(orchestrate_on(first, 0, backend, session).code ==
         megacu::status_code::ok);
  assert(orchestrate_on(second, 1, backend, session).code ==
         megacu::status_code::ok);

  free_device(second);
  free_device(first);
  return 0;
}
