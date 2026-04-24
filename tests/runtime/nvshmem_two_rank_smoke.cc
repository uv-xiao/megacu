#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce.h"

// The CUDA 12.8 Debian package ships a host library that exports these C API
// symbols, while its CMake package references a missing device archive. Keep
// this smoke on the exported host ABI until the device package is available.
extern "C" {
int nvshmemx_hostlib_init_attr(unsigned int flags, void *attr);
void nvshmemx_hostlib_finalize();
int nvshmem_my_pe();
int nvshmem_n_pes();
void *nvshmem_malloc(std::size_t size);
void nvshmem_free(void *ptr);
void nvshmem_barrier_all();
}

namespace {

constexpr int kSkipTest = 77;
constexpr std::size_t kBytes = 256;

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
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
  assert(nvshmemx_hostlib_init_attr(0, nullptr) == 0);

  int pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  if (npes != 2) {
    nvshmemx_hostlib_finalize();
    return kSkipTest;
  }

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    nvshmemx_hostlib_finalize();
    return kSkipTest;
  }

  int device = pe % device_count;
  require_cuda(cudaSetDevice(device));

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  void *scratch = nullptr;
  require_cuda(cudaMalloc(&a, kBytes));
  require_cuda(cudaMalloc(&b, kBytes));
  require_cuda(cudaMalloc(&c, kBytes));
  require_cuda(cudaMalloc(&scratch, kBytes));

  void *partial = nvshmem_malloc(kBytes);
  void *events = nvshmem_malloc(kBytes);
  assert(partial != nullptr);
  assert(events != nullptr);

  megacu::backend_id backend{1};
  megacu::session_id session{7};

  gemm_ar_workspace workspace{
      .a = {
          .data = a,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
      .b = {
          .data = b,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
      .partial = {
          .buffer = {
              .data = partial,
              .bytes = static_cast<std::int64_t>(kBytes),
              .backend = backend,
              .session = session},
          .type = megacu::dtype::f32},
      .c = {
          .data = c,
          .bytes = static_cast<std::int64_t>(kBytes),
          .type = megacu::dtype::f32},
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
      .team_my_pe = pe,
      .team_n_pes = npes,
      .world_my_pe = pe,
      .world_n_pes = npes,
      .cuda_device_ordinal = device,
      .backend = backend,
      .session = session};

  auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, event_storage, launch, team, small_problem());
  assert(status.code == megacu::status_code::ok);

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(partial);

  require_cuda(cudaFree(scratch));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  nvshmemx_hostlib_finalize();
  return 0;
}
