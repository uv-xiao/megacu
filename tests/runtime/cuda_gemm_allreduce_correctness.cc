#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"

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

megacu::status double_local_sum(
    megacu::nvshmem::team_view,
    void *dest,
    void const *src,
    std::int64_t elements,
    void *stream) {
  std::vector<float> host(static_cast<std::size_t>(elements));
  require_cuda(cudaMemcpyAsync(
      host.data(),
      src,
      host.size() * sizeof(float),
      cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(stream)));
  require_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  for (float &value : host) {
    value *= 2.0f;
  }
  require_cuda(cudaMemcpyAsync(
      dest,
      host.data(),
      host.size() * sizeof(float),
      cudaMemcpyHostToDevice,
      static_cast<cudaStream_t>(stream)));
  return {};
}

float expected(std::vector<float> const &a,
               std::vector<float> const &b,
               int row,
               int col) {
  float value = 0.0f;
  for (int kk = 0; kk < kK; ++kk) {
    value += a[row * kK + kk] * b[kk * kN + col];
  }
  return 2.0f * value;
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
  void *scratch = nullptr;
  void *events = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&c, kCBytes));
  require_cuda(cudaMalloc(&partial, kPartialBytes));
  require_cuda(cudaMalloc(&scratch, 128));
  require_cuda(cudaMalloc(&events, 128));

  require_cuda(cudaMemcpyAsync(
      a, host_a.data(), kABytes, cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(
      b, host_b.data(), kBBytes, cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemsetAsync(c, 0, kCBytes, stream));

  megacu::backend_id backend{1};
  megacu::session_id session{7};
  gemm_ar_comm_ops ops{.sum_reduce_f32 = double_local_sum};

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
      .team_my_pe = 0,
      .team_n_pes = 2,
      .world_my_pe = 0,
      .world_n_pes = 2,
      .cuda_device_ordinal = device,
      .backend = backend,
      .session = session};

  auto status = cuda_nvshmem_gemm_allreduce_phased_orchestrate(
      workspace,
      event_storage,
      launch,
      team,
      gemm_ar_problem{.m = kM, .n = kN, .k = kK, .tile_m = kM, .tile_n = kN});
  assert(status.code == megacu::status_code::ok);

  require_cuda(cudaMemcpyAsync(
      host_c.data(), c, kCBytes, cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      float want = expected(host_a, host_b, row, col);
      float got = host_c[row * kN + col];
      assert(std::fabs(got - want) < 1.0e-4f);
    }
  }

  require_cuda(cudaFree(events));
  require_cuda(cudaFree(scratch));
  require_cuda(cudaFree(partial));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));
  return 0;
}
