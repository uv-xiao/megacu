#include <cuda_runtime.h>
#include <nvshmem_host.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"
#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

namespace {

constexpr int kSkipTest = 77;
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

gemm_ar_problem correctness_problem() {
  return {
      .m = kM,
      .n = kN,
      .k = kK,
      .tile_m = kM,
      .tile_n = kN};
}

void reset_outputs(void *c, void *partial, void *events, cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(c, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(partial, 0, kPartialBytes, stream));
  require_cuda(cudaMemsetAsync(events, 0, 128, stream));
  require_cuda(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();
}

void check_expected(char const *label, int pe, void *c, cudaStream_t stream) {
  std::vector<float> host_c(kM * kN, 0.0f);
  require_cuda(cudaMemcpyAsync(
      host_c.data(), c, kCBytes, cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  for (float value : host_c) {
    if (std::fabs(value - 12.0f) >= 1.0e-4f) {
      std::fprintf(stderr, "%s pe=%d values:", label, pe);
      for (float item : host_c) {
        std::fprintf(stderr, " %.6f", item);
      }
      std::fprintf(stderr, "\n");
    }
    assert(std::fabs(value - 12.0f) < 1.0e-4f);
  }
}

}  // namespace

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
  golden_gemm_ar_team golden_team{
      .team = nullptr,
      .team_my_pe = pe,
      .team_n_pes = npes,
      .world_my_pe = pe,
      .world_n_pes = npes,
      .cuda_device_ordinal = device};
  golden_gemm_ar_problem golden_problem{
      .m = kM,
      .n = kN,
      .k = kK,
      .tile_m = 1,
      .tile_n = 2,
      .compute_ctas = 2,
      .comm_ctas = 1};

  megacu::backend_id backend{1};
  megacu::session_id session{7};

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
      .team = nullptr,
      .team_my_pe = pe,
      .team_n_pes = npes,
      .world_my_pe = pe,
      .world_n_pes = npes,
      .cuda_device_ordinal = device,
      .backend = backend,
      .session = session};

  if (std::strcmp(mode, "golden_phased") == 0) {
    auto golden_status = golden_phased_multi_card_gemm_allreduce_f32(
        golden_workspace, golden_launch, golden_team, golden_problem);
    if (golden_status.code != golden_status_code::ok) {
      std::fprintf(
          stderr,
          "golden phased failed: code=%d detail=%u message=%s\n",
          static_cast<int>(golden_status.code),
          golden_status.detail,
          golden_status.message);
    }
    assert(golden_status.code == golden_status_code::ok);
    check_expected("golden phased", pe, c, stream);
  } else if (std::strcmp(mode, "golden_overlap") == 0) {
    auto golden_status = golden_overlap_multi_card_gemm_allreduce_f32(
        golden_workspace, golden_launch, golden_team, golden_problem);
    assert(golden_status.code == golden_status_code::ok);
    check_expected("golden overlap", pe, c, stream);
  } else if (std::strcmp(mode, "megacu_phased") == 0) {
    auto status = cuda_nvshmem_gemm_allreduce_phased_orchestrate(
        workspace, event_storage, launch, team, correctness_problem());
    assert(status.code == megacu::status_code::ok);
    check_expected("megacu phased", pe, c, stream);
  } else if (std::strcmp(mode, "megacu_overlap") == 0) {
    auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
        workspace, event_storage, launch, team, correctness_problem());
    assert(status.code == megacu::status_code::ok);
    check_expected("megacu overlap", pe, c, stream);
  } else {
    assert(false && "unknown nvshmem smoke mode");
  }
  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(partial);

  require_cuda(cudaFree(scratch));
  require_cuda(cudaFree(c));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  nvshmem_finalize();
  return 0;
}
