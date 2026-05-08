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
constexpr int kK = 4;
constexpr int kLocalK = kK / 2;
constexpr std::size_t kABytes = kM * kK * sizeof(float);
constexpr std::size_t kBLocalBytes = kLocalK * kN * sizeof(float);
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

void check_expected(char const *label, int pe, void *out,
                    cudaStream_t stream) {
  std::vector<float> host_out(kM * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_out.data(), out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto got = host_out[row * kN + col];
      auto want = 6.0F;
      if (std::fabs(got - want) >= 1.0e-4F) {
        std::fprintf(stderr,
                     "%s pe=%d mismatch row=%d col=%d got=%.6f want=%.6f\n",
                     label, pe, row, col, got, want);
      }
      assert(std::fabs(got - want) < 1.0e-4F);
    }
  }
}

void run_case(char const *label,
              megacu::status (*fn)(ag_gemm_driver, float const *,
                                   float const *, float *, float *, void *,
                                   ag_gemm_problem),
              ag_gemm_driver driver, void const *a, void const *b,
              void *gathered_b, void *out, void *events, cudaStream_t stream,
              int pe) {
  reset_outputs(out, gathered_b, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b),
                   static_cast<float *>(gathered_b), static_cast<float *>(out),
                   events, correctness_problem());
  assert(status.code == megacu::status_code::ok);
  check_expected(label, pe, out, stream);
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
  void *gathered_b = nullptr;
  void *out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&gathered_b, kGatheredBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));

  void *b = nvshmem_malloc(kBLocalBytes);
  void *events = nvshmem_malloc(128);
  assert(b != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, 1.0F);
  std::vector<float> host_b(kLocalK * kN, static_cast<float>(pe + 1));
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
                               cudaMemcpyHostToDevice, stream));
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
             driver, a, b, gathered_b, out, events, stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("ag-gemm seeded-orch", cuda_nvshmem_allgather_gemm_seeded_orch,
             driver, a, b, gathered_b, out, events, stream, pe);
  } else {
    assert(false && "unknown AG-GEMM two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(b);

  require_cuda(cudaFree(out));
  require_cuda(cudaFree(gathered_b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  finalize_backend();
  return 0;
}
