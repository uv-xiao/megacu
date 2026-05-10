#include "examples/cuda_nvshmem/gemm_reduce_scatter/common/gemm_reduce_scatter.h"

#include <cuda_runtime.h>
#include <nvshmem_host.h>
#if defined(MEGACU_GEMM_RS_SMOKE_USE_MPI_BOOTSTRAP)
#include <mpi.h>
#endif
#if defined(MEGACU_GEMM_RS_SMOKE_USE_MPI_BOOTSTRAP) ||                      \
    defined(MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP)
#include <nvshmemx.h>
#endif

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
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

char const *first_env(std::initializer_list<char const *> names) {
  for (auto *name : names) {
    auto *value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
      return value;
    }
  }
  return nullptr;
}

int optional_env(std::initializer_list<char const *> names, int fallback) {
  auto *value = first_env(names);
  return value == nullptr ? fallback : std::atoi(value);
}

int required_env(std::initializer_list<char const *> names) {
  auto *value = first_env(names);
  assert(value != nullptr);
  return std::atoi(value);
}

int hex_value(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  value = static_cast<char>(std::tolower(value));
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  assert(false && "invalid hex digit");
  return 0;
}

void decode_hex(char const *hex, unsigned char *bytes, std::size_t byte_count) {
  assert(hex != nullptr);
  assert(std::strlen(hex) == byte_count * 2);
  for (std::size_t index = 0; index < byte_count; ++index) {
    bytes[index] =
        static_cast<unsigned char>((hex_value(hex[index * 2]) << 4) |
                                   hex_value(hex[index * 2 + 1]));
  }
}

std::string encode_hex(void const *data, std::size_t byte_count) {
  auto const *bytes = static_cast<unsigned char const *>(data);
  char const digits[] = "0123456789abcdef";
  std::string hex;
  hex.resize(byte_count * 2);
  for (std::size_t index = 0; index < byte_count; ++index) {
    hex[index * 2] = digits[bytes[index] >> 4];
    hex[index * 2 + 1] = digits[bytes[index] & 0x0F];
  }
  return hex;
}

struct launch_env {
  int device = -1;
  bool skip = false;
};

int print_unique_id() {
#if defined(MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP)
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto status = nvshmemx_get_uniqueid(&id);
  assert(status == 0);
  auto hex = encode_hex(&id, sizeof(id));
  std::printf("%s\n", hex.c_str());
  return 0;
#else
  return 1;
#endif
}

char const *prepare_uid_root_mode() {
#if defined(MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP)
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto status = nvshmemx_get_uniqueid(&id);
  assert(status == 0);
  auto hex = encode_hex(&id, sizeof(id));
  std::printf("%s\n", hex.c_str());
  std::fflush(stdout);
  (void)std::getchar();
  auto set_status =
      setenv("MEGACU_NVSHMEM_UNIQUEID_HEX", hex.c_str(), 1);
  assert(set_status == 0);
  return std::getenv("MEGACU_NVSHMEM_UNIQUEID_HEX");
#else
  return nullptr;
#endif
}

launch_env initialize_backend(int *argc, char ***argv) {
#if defined(MEGACU_GEMM_RS_SMOKE_USE_MPI_BOOTSTRAP)
  MPI_Init(argc, argv);
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    return {.device = -1, .skip = true};
  }

  auto local_rank =
      optional_env({"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                    "PMI_LOCAL_RANK"},
                   mpi_rank);
  auto device = local_rank % device_count;
  require_cuda(cudaSetDevice(device));

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  auto init_status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  assert(init_status == 0);
  return {.device = device, .skip = false};
#elif defined(MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP)
  (void)argc;
  (void)argv;
  auto rank = required_env({"RANK"});
  auto world_size = required_env({"WORLD_SIZE"});
  auto local_rank = required_env({"LOCAL_RANK"});

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    return {.device = -1, .skip = true};
  }

  auto device = local_rank % device_count;
  require_cuda(cudaSetDevice(device));

  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  decode_hex(first_env({"MEGACU_NVSHMEM_UNIQUEID_HEX"}),
             reinterpret_cast<unsigned char *>(&id), sizeof(id));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &id, &attr);
  auto init_status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  assert(init_status == 0);
  return {.device = device, .skip = false};
#else
  (void)argc;
  (void)argv;
  nvshmem_init();
  return {};
#endif
}

void finalize_backend() {
  nvshmem_finalize();
#if defined(MEGACU_GEMM_RS_SMOKE_USE_MPI_BOOTSTRAP)
  MPI_Finalize();
#endif
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

void reset_local_outputs(void *out, void *partial, cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(partial, 0, kCBytes, stream));
}

void check_against_references(char const *label, int pe, void *candidate,
                              void *golden, void *baseline,
                              cudaStream_t stream) {
  std::vector<float> host_candidate(kM * kN, 0.0F);
  std::vector<float> host_golden(kM * kN, 0.0F);
  std::vector<float> host_baseline(kM * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_candidate.data(), candidate, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaMemcpyAsync(host_golden.data(), golden, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaMemcpyAsync(host_baseline.data(), baseline, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto const index = row * kN + col;
      auto got = host_candidate[index];
      auto golden_want = host_golden[index];
      auto baseline_want = host_baseline[index];
      if (std::fabs(got - golden_want) >= 1.0e-4F ||
          std::fabs(got - baseline_want) >= 1.0e-4F) {
        std::fprintf(stderr,
                     "%s pe=%d mismatch row=%d col=%d got=%.6f golden=%.6f "
                     "baseline=%.6f\n",
                     label, pe, row, col, got, golden_want, baseline_want);
      }
      assert(std::fabs(got - golden_want) < 1.0e-4F);
      assert(std::fabs(got - baseline_want) < 1.0e-4F);
    }
  }
}

void run_case(char const *label,
              megacu::status (*fn)(gemm_rs_driver, float const *,
                                    float const *, float *, float *, void *,
                                    gemm_rs_problem),
              gemm_rs_driver driver, void const *a, void const *b,
              void const *reference_a, void *partial, void *out,
              void *events, void *golden_partial, void *golden_out,
              void *baseline_partial, void *baseline_out,
              cudaStream_t stream, int pe) {
  reset_local_outputs(golden_out, golden_partial, stream);
  auto golden_status = golden_cuda_nvshmem_gemm_reduce_scatter(
      driver, static_cast<float const *>(reference_a),
      static_cast<float const *>(b), static_cast<float *>(golden_partial),
      static_cast<float *>(golden_out), correctness_problem());
  assert(golden_status.code == megacu::status_code::ok);

  reset_local_outputs(baseline_out, baseline_partial, stream);
  auto baseline_status = baseline_cuda_nvshmem_gemm_reduce_scatter(
      driver, static_cast<float const *>(reference_a),
      static_cast<float const *>(b), static_cast<float *>(baseline_partial),
      static_cast<float *>(baseline_out), correctness_problem());
  assert(baseline_status.code == megacu::status_code::ok);

  reset_outputs(out, partial, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b), static_cast<float *>(partial),
                   static_cast<float *>(out), events, correctness_problem());
  assert(status.code == megacu::status_code::ok);
  check_against_references(label, pe, out, golden_out, baseline_out, stream);
}

} // namespace

int main(int argc, char **argv) {
  char const *mode = argc > 1 ? argv[1] : "";
  if (std::strcmp(mode, "--print-uid") == 0) {
    return print_unique_id();
  }
  if (std::strcmp(mode, "--uid-bootstrap-root") == 0) {
    assert(argc > 2);
    (void)prepare_uid_root_mode();
    mode = argv[2];
  }

  auto launch = initialize_backend(&argc, &argv);
  if (launch.skip) {
    finalize_backend();
    return kSkipTest;
  }

  int pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  if (npes != 2) {
    finalize_backend();
    return kSkipTest;
  }

  int device = launch.device;
  if (device < 0) {
    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
      finalize_backend();
      return kSkipTest;
    }
    device = pe % device_count;
    require_cuda(cudaSetDevice(device));
  }

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  void *a = nullptr;
  void *b = nullptr;
  void *reference_a = nullptr;
  void *out = nullptr;
  void *golden_out = nullptr;
  void *baseline_out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&reference_a, kABytes));
  require_cuda(cudaMalloc(&out, kCBytes));
  require_cuda(cudaMalloc(&golden_out, kCBytes));
  require_cuda(cudaMalloc(&baseline_out, kCBytes));

  void *partial = nvshmem_malloc(kCBytes);
  void *events = nvshmem_malloc(128);
  void *golden_partial = nullptr;
  void *baseline_partial = nullptr;
  require_cuda(cudaMalloc(&golden_partial, kCBytes));
  require_cuda(cudaMalloc(&baseline_partial, kCBytes));
  assert(partial != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, static_cast<float>(pe + 1));
  std::vector<float> reference_host_a(kM * kK, 3.0F);
  std::vector<float> host_b(kK * kN, 1.0F);
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(reference_a, reference_host_a.data(), kABytes,
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
             driver, a, b, reference_a, partial, out, events, golden_partial,
             golden_out, baseline_partial, baseline_out, stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("gemm-rs seeded-orch",
             cuda_nvshmem_gemm_reduce_scatter_seeded_orch, driver, a, b,
             reference_a, partial, out, events, golden_partial, golden_out,
             baseline_partial, baseline_out, stream, pe);
  } else {
    assert(false && "unknown GEMM-RS two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(partial);

  require_cuda(cudaFree(baseline_partial));
  require_cuda(cudaFree(golden_partial));
  require_cuda(cudaFree(baseline_out));
  require_cuda(cudaFree(golden_out));
  require_cuda(cudaFree(out));
  require_cuda(cudaFree(reference_a));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  finalize_backend();
  return 0;
}
