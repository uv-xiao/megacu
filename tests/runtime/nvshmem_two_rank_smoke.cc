#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"

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
int nvshmem_float_sum_reduce(
    int team,
    float *dest,
    float const *src,
    std::size_t nreduce);
}

namespace {

constexpr int kSkipTest = 77;
constexpr int kM = 2;
constexpr int kN = 3;
constexpr int kK = 4;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBBytes = kK * kN * sizeof(float);
constexpr std::size_t kCBytes = kM * kN * sizeof(float);
constexpr std::size_t kPartialBytes = 2 * kCBytes;
constexpr int kNvshmemTeamWorld = 0;

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

gemm_ar_problem correctness_problem() {
  return {
      .m = kM,
      .n = kN,
      .k = kK,
      .tile_m = kM,
      .tile_n = kN};
}

megacu::status nvshmem_sum_reduce_f32(
    megacu::nvshmem::team_view,
    void *dest,
    void const *src,
    std::int64_t elements,
    void *stream) {
  require_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  auto rc = nvshmem_float_sum_reduce(
      kNvshmemTeamWorld,
      static_cast<float *>(dest),
      static_cast<float const *>(src),
      static_cast<std::size_t>(elements));
  if (rc != 0) {
    return {megacu::status_code::backend_error, 1, "nvshmem sum reduce failed"};
  }
  nvshmem_barrier_all();
  return {};
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
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&c, kCBytes));
  require_cuda(cudaMalloc(&scratch, 128));

  void *partial = nvshmem_malloc(kPartialBytes);
  void *events = nvshmem_malloc(128);
  assert(partial != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, static_cast<float>(pe + 1));
  std::vector<float> host_b(kK * kN, 1.0f);
  std::vector<float> host_c(kM * kN, 0.0f);
  require_cuda(cudaMemcpyAsync(
      a, host_a.data(), kABytes, cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(
      b, host_b.data(), kBBytes, cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemsetAsync(c, 0, kCBytes, stream));

  megacu::backend_id backend{1};
  megacu::session_id session{7};
  gemm_ar_comm_ops ops{.sum_reduce_f32 = nvshmem_sum_reduce_f32};

  gemm_ar_workspace workspace{
      .a = {
          .data = a,
          .bytes = static_cast<std::int64_t>(kABytes),
          .type = megacu::dtype::f32},
      .b = {
          .data = b,
          .bytes = static_cast<std::int64_t>(kBBytes),
          .type = megacu::dtype::f32},
      .partial = {
          .buffer = {
              .data = partial,
              .bytes = static_cast<std::int64_t>(kPartialBytes),
              .backend = backend,
              .session = session},
          .type = megacu::dtype::f32},
      .c = {
          .data = c,
          .bytes = static_cast<std::int64_t>(kCBytes),
          .type = megacu::dtype::f32},
      .scratch = megacu::span<std::byte>{
          static_cast<std::byte *>(scratch),
          128}};

  megacu::event_storage_view event_storage{
      .buffer = {
          .data = events,
          .bytes = 128,
          .backend = backend,
          .session = session}};

  megacu::cuda::launch_view launch{
      .stream = stream,
      .device_ordinal = device};
  megacu::nvshmem::team_view team{
      .team = &ops,
      .team_my_pe = pe,
      .team_n_pes = npes,
      .world_my_pe = pe,
      .world_n_pes = npes,
      .cuda_device_ordinal = device,
      .backend = backend,
      .session = session};

  auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, event_storage, launch, team, correctness_problem());
  assert(status.code == megacu::status_code::ok);

  require_cuda(cudaMemcpyAsync(
      host_c.data(), c, kCBytes, cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  for (float value : host_c) {
    assert(std::fabs(value - 12.0f) < 1.0e-4f);
  }

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
