#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cuda_runtime.h>
#include <nvshmem_host.h>
#if defined(MEGACU_AG_GEMM_SMOKE_USE_MPI_BOOTSTRAP)
#include <mpi.h>
#endif
#if defined(MEGACU_AG_GEMM_SMOKE_USE_MPI_BOOTSTRAP) ||                      \
    defined(MEGACU_AG_GEMM_SMOKE_USE_UID_BOOTSTRAP)
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
constexpr int kM = 3;
constexpr int kN = 4;
constexpr int kK = 5;
constexpr int kMaxLocalK = 3;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBLocalBytes = kMaxLocalK * kN * sizeof(float);
constexpr std::size_t kGatheredBBytes = kK * kN * sizeof(float);
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

int k_slice_begin(int global_k, int n_pes, int pe) {
  auto base = global_k / n_pes;
  auto extra = global_k % n_pes;
  return pe * base + (pe < extra ? pe : extra);
}

int k_slice_count(int global_k, int n_pes, int pe) {
  auto base = global_k / n_pes;
  auto extra = global_k % n_pes;
  return base + (pe < extra ? 1 : 0);
}

int owner_for_k(int global_k, int n_pes, int kk) {
  for (int owner = 0; owner < n_pes; ++owner) {
    auto begin = k_slice_begin(global_k, n_pes, owner);
    auto count = k_slice_count(global_k, n_pes, owner);
    if (kk >= begin && kk < begin + count) {
      return owner;
    }
  }
  return n_pes - 1;
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
#if defined(MEGACU_AG_GEMM_SMOKE_USE_UID_BOOTSTRAP)
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
#if defined(MEGACU_AG_GEMM_SMOKE_USE_UID_BOOTSTRAP)
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto status = nvshmemx_get_uniqueid(&id);
  assert(status == 0);
  auto hex = encode_hex(&id, sizeof(id));
  std::printf("%s\n", hex.c_str());
  std::fflush(stdout);
  (void)std::getchar();
  auto set_status = setenv("MEGACU_NVSHMEM_UNIQUEID_HEX", hex.c_str(), 1);
  assert(set_status == 0);
  return std::getenv("MEGACU_NVSHMEM_UNIQUEID_HEX");
#else
  return nullptr;
#endif
}

launch_env initialize_backend(int *argc, char ***argv) {
#if defined(MEGACU_AG_GEMM_SMOKE_USE_MPI_BOOTSTRAP)
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
#elif defined(MEGACU_AG_GEMM_SMOKE_USE_UID_BOOTSTRAP)
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
#if defined(MEGACU_AG_GEMM_SMOKE_USE_MPI_BOOTSTRAP)
  MPI_Finalize();
#endif
}

ag_gemm_problem correctness_problem() {
  return {.m = kM, .n = kN, .k = kK, .tile_m = 1, .tile_n = 2, .tile_k = 1};
}

void reset_outputs(void *out, void *gathered_b, void *events,
                   cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(gathered_b, 0, kGatheredBBytes, stream));
  require_cuda(cudaMemsetAsync(events, 0, 128, stream));
  require_cuda(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();
}

void reset_local_outputs(void *out, void *gathered_b, cudaStream_t stream) {
  require_cuda(cudaMemsetAsync(out, 0, kCBytes, stream));
  require_cuda(cudaMemsetAsync(gathered_b, 0, kGatheredBBytes, stream));
}

void check_against_references(char const *label, int pe, void *candidate_out,
                              void *golden_out, void *baseline_out,
                              void *candidate_gathered_b,
                              void *golden_gathered_b,
                              void *baseline_gathered_b,
                              cudaStream_t stream) {
  std::vector<float> host_candidate(kM * kN, 0.0F);
  std::vector<float> host_golden(kM * kN, 0.0F);
  std::vector<float> host_baseline(kM * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_candidate.data(), candidate_out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaMemcpyAsync(host_golden.data(), golden_out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaMemcpyAsync(host_baseline.data(), baseline_out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));

  std::vector<float> host_gathered_candidate(kK * kN, 0.0F);
  std::vector<float> host_gathered_golden(kK * kN, 0.0F);
  std::vector<float> host_gathered_baseline(kK * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_gathered_candidate.data(),
                               candidate_gathered_b, kGatheredBBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaMemcpyAsync(host_gathered_golden.data(), golden_gathered_b,
                               kGatheredBBytes, cudaMemcpyDeviceToHost,
                               stream));
  require_cuda(cudaMemcpyAsync(host_gathered_baseline.data(),
                               baseline_gathered_b, kGatheredBBytes,
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
                     "%s pe=%d output mismatch row=%d col=%d got=%.6f "
                     "golden=%.6f baseline=%.6f\n",
                     label, pe, row, col, got, golden_want, baseline_want);
      }
      assert(std::fabs(got - golden_want) < 1.0e-4F);
      assert(std::fabs(got - baseline_want) < 1.0e-4F);
    }
  }

  for (int kk = 0; kk < kK; ++kk) {
    for (int col = 0; col < kN; ++col) {
      auto const index = kk * kN + col;
      auto got = host_gathered_candidate[index];
      auto golden_want = host_gathered_golden[index];
      auto baseline_want = host_gathered_baseline[index];
      if (std::fabs(got - golden_want) >= 1.0e-4F ||
          std::fabs(got - baseline_want) >= 1.0e-4F) {
        std::fprintf(stderr,
                     "%s pe=%d gathered_b mismatch kk=%d col=%d got=%.6f "
                     "golden=%.6f baseline=%.6f\n",
                     label, pe, kk, col, got, golden_want, baseline_want);
      }
      assert(std::fabs(got - golden_want) < 1.0e-4F);
      assert(std::fabs(got - baseline_want) < 1.0e-4F);
    }
  }
}

void run_case(char const *label,
              megacu::status (*fn)(ag_gemm_driver, float const *,
                                    float const *, float *, float *, void *,
                                    ag_gemm_problem),
              ag_gemm_driver driver, void const *a, void const *b,
              void const *reference_b, void *gathered_b, void *out,
              void *events, void *golden_gathered_b, void *golden_out,
              void *baseline_gathered_b, void *baseline_out,
              cudaStream_t stream, int pe) {
  reset_local_outputs(golden_out, golden_gathered_b, stream);
  auto golden_status = golden_cuda_nvshmem_allgather_gemm(
      driver, static_cast<float const *>(a),
      static_cast<float const *>(reference_b),
      static_cast<float *>(golden_gathered_b), static_cast<float *>(golden_out),
      correctness_problem());
  assert(golden_status.code == megacu::status_code::ok);

  reset_local_outputs(baseline_out, baseline_gathered_b, stream);
  auto baseline_status = baseline_cuda_nvshmem_allgather_gemm(
      driver, static_cast<float const *>(a),
      static_cast<float const *>(reference_b),
      static_cast<float *>(baseline_gathered_b),
      static_cast<float *>(baseline_out), correctness_problem());
  assert(baseline_status.code == megacu::status_code::ok);

  reset_outputs(out, gathered_b, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b),
                   static_cast<float *>(gathered_b), static_cast<float *>(out),
                   events, correctness_problem());
  assert(status.code == megacu::status_code::ok);
  check_against_references(label, pe, out, golden_out, baseline_out, gathered_b,
                           golden_gathered_b, baseline_gathered_b, stream);
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

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    finalize_backend();
    return kSkipTest;
  }

  int device = launch.device;
  if (device < 0) {
    device = pe % device_count;
    require_cuda(cudaSetDevice(device));
  }

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream));

  void *a = nullptr;
  void *reference_b = nullptr;
  void *gathered_b = nullptr;
  void *out = nullptr;
  void *golden_gathered_b = nullptr;
  void *golden_out = nullptr;
  void *baseline_gathered_b = nullptr;
  void *baseline_out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&reference_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&gathered_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));
  require_cuda(cudaMalloc(&golden_gathered_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&golden_out, kCBytes));
  require_cuda(cudaMalloc(&baseline_gathered_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&baseline_out, kCBytes));

  void *b = nvshmem_malloc(kBLocalBytes);
  void *events = nvshmem_malloc(128);
  assert(b != nullptr);
  assert(events != nullptr);

  auto local_k_count = k_slice_count(kK, npes, pe);
  std::vector<float> host_a(kM * kK, 1.0F);
  std::vector<float> host_b(kMaxLocalK * kN, 0.0F);
  for (int local_k = 0; local_k < local_k_count; ++local_k) {
    for (int col = 0; col < kN; ++col) {
      host_b[local_k * kN + col] = static_cast<float>(pe + 1);
    }
  }
  std::vector<float> reference_host_b(kK * kN, 0.0F);
  for (int kk = 0; kk < kK; ++kk) {
    auto owner_value = static_cast<float>(owner_for_k(kK, npes, kk) + 1);
    for (int col = 0; col < kN; ++col) {
      reference_host_b[kk * kN + col] = owner_value;
    }
  }
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(reference_b, reference_host_b.data(),
                               kGatheredBBytes, cudaMemcpyHostToDevice,
                               stream));
  require_cuda(cudaMemcpyAsync(b, host_b.data(), kBLocalBytes,
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  ag_gemm_driver driver{
      .launch = {.stream = stream, .device_ordinal = device},
      .team = {.team = nullptr,
               .team_my_pe = pe,
               .team_n_pes = npes,
               .world_my_pe = pe,
               .world_n_pes = npes,
               .cuda_device_ordinal = device}};

  if (std::strcmp(mode, "megacu_host_orch") == 0) {
    run_case("ag-gemm host-orch", cuda_nvshmem_allgather_gemm_host_orch,
             driver, a, b, reference_b, gathered_b, out, events,
             golden_gathered_b, golden_out, baseline_gathered_b, baseline_out,
             stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("ag-gemm seeded-orch", cuda_nvshmem_allgather_gemm_seeded_orch,
             driver, a, b, reference_b, gathered_b, out, events,
             golden_gathered_b, golden_out, baseline_gathered_b, baseline_out,
             stream, pe);
  } else {
    assert(false && "unknown AG-GEMM two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(b);

  require_cuda(cudaFree(baseline_out));
  require_cuda(cudaFree(baseline_gathered_b));
  require_cuda(cudaFree(golden_out));
  require_cuda(cudaFree(golden_gathered_b));
  require_cuda(cudaFree(out));
  require_cuda(cudaFree(gathered_b));
  require_cuda(cudaFree(reference_b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  finalize_backend();
  return 0;
}
