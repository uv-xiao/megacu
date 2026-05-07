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

struct launch_env {
  int device = -1;
  bool skip = false;
};

int print_unique_id() {
#if defined(MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP)
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto status = nvshmemx_get_uniqueid(&id);
  assert(status == 0);
  auto const *bytes = reinterpret_cast<unsigned char const *>(&id);
  for (std::size_t index = 0; index < sizeof(id); ++index) {
    std::printf("%02x", static_cast<unsigned>(bytes[index]));
  }
  std::printf("\n");
  return 0;
#else
  return 1;
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

void check_expected(char const *label, int pe, void *out,
                    cudaStream_t stream) {
  std::vector<float> host_out(kM * kN, 0.0F);
  require_cuda(cudaMemcpyAsync(host_out.data(), out, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto want = row % 2 == pe ? 12.0F : 0.0F;
      auto got = host_out[row * kN + col];
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
              megacu::status (*fn)(gemm_rs_driver, float const *,
                                   float const *, float *, float *, void *,
                                   gemm_rs_problem),
              gemm_rs_driver driver, void const *a, void const *b,
              void *partial, void *out, void *events, cudaStream_t stream,
              int pe) {
  reset_outputs(out, partial, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b), static_cast<float *>(partial),
                   static_cast<float *>(out), events, correctness_problem());
  assert(status.code == megacu::status_code::ok);
  check_expected(label, pe, out, stream);
}

} // namespace

int main(int argc, char **argv) {
  char const *mode = argc > 1 ? argv[1] : "";
  if (std::strcmp(mode, "--print-uid") == 0) {
    return print_unique_id();
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
  void *out = nullptr;
  require_cuda(cudaMalloc(&a, kABytes));
  require_cuda(cudaMalloc(&b, kBBytes));
  require_cuda(cudaMalloc(&out, kCBytes));

  void *partial = nvshmem_malloc(kCBytes);
  void *events = nvshmem_malloc(128);
  assert(partial != nullptr);
  assert(events != nullptr);

  std::vector<float> host_a(kM * kK, static_cast<float>(pe + 1));
  std::vector<float> host_b(kK * kN, 1.0F);
  require_cuda(cudaMemcpyAsync(a, host_a.data(), kABytes,
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
             driver, a, b, partial, out, events, stream, pe);
  } else if (std::strcmp(mode, "megacu_seeded_orch") == 0) {
    run_case("gemm-rs seeded-orch",
             cuda_nvshmem_gemm_reduce_scatter_seeded_orch, driver, a, b,
             partial, out, events, stream, pe);
  } else {
    assert(false && "unknown GEMM-RS two-rank mode");
  }

  nvshmem_barrier_all();
  nvshmem_free(events);
  nvshmem_free(partial);

  require_cuda(cudaFree(out));
  require_cuda(cudaFree(b));
  require_cuda(cudaFree(a));
  require_cuda(cudaStreamDestroy(stream));

  finalize_backend();
  return 0;
}
