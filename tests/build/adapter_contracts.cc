#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <string>

#include <megacu/adapters/mpi_cuda_nvshmem.h>
#include <megacu/adapters/torch_cuda_nvshmem.h>

namespace {

int fail(std::string const &message) {
  std::cerr << "adapter contract failed: " << message << '\n';
  return 1;
}

int expect(bool condition, std::string const &message) {
  return condition ? 0 : fail(message);
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

int parse_required(std::initializer_list<char const *> names,
                   char const *label) {
  auto *value = first_env(names);
  if (value == nullptr) {
    std::cerr << "missing " << label << " environment fact\n";
    std::exit(1);
  }
  return std::stoi(value);
}

int parse_optional(std::initializer_list<char const *> names,
                   int fallback) {
  auto *value = first_env(names);
  return value == nullptr ? fallback : std::stoi(value);
}

template <class Facts>
int check_driver(megacu::cuda_nvshmem::driver_view driver,
                 Facts facts,
                 void *stream) {
  if (auto rc = expect(driver.launch.stream == stream, "stream mismatch")) {
    return rc;
  }
  if (auto rc = expect(driver.launch.device_ordinal == facts.local_rank,
                       "device ordinal mismatch")) {
    return rc;
  }
  if (auto rc = expect(driver.team.team_my_pe == facts.rank,
                       "team rank mismatch")) {
    return rc;
  }
  if (auto rc = expect(driver.team.team_n_pes == facts.world_size,
                       "team size mismatch")) {
    return rc;
  }
  if (auto rc = expect(driver.team.world_my_pe == facts.rank,
                       "world rank mismatch")) {
    return rc;
  }
  if (auto rc = expect(driver.team.world_n_pes == facts.world_size,
                       "world size mismatch")) {
    return rc;
  }
  return expect(driver.team.cuda_device_ordinal == facts.local_rank,
                "team device ordinal mismatch");
}

int check_local_contract() {
  megacu::adapters::mpi_cuda_nvshmem::launch_facts mpi_facts{
      .rank = 2, .world_size = 4, .local_rank = 1};
  auto mpi = megacu::adapters::mpi_cuda_nvshmem::make_driver(
      mpi_facts, reinterpret_cast<void *>(0x1));
  if (auto rc = check_driver(mpi, mpi_facts,
                             reinterpret_cast<void *>(0x1))) {
    return rc;
  }

  megacu::adapters::torch_cuda_nvshmem::launch_facts torch_facts{
      .rank = 3, .world_size = 8, .local_rank = 0};
  auto torch = megacu::adapters::torch_cuda_nvshmem::make_driver(
      torch_facts);
  return check_driver(torch, torch_facts, nullptr);
}

int check_mpi_env_contract() {
  megacu::adapters::mpi_cuda_nvshmem::launch_facts facts{
      .rank = parse_required({"OMPI_COMM_WORLD_RANK", "PMI_RANK"},
                             "MPI rank"),
      .world_size = parse_required({"OMPI_COMM_WORLD_SIZE", "PMI_SIZE"},
                                   "MPI world size"),
      .local_rank = parse_required({"OMPI_COMM_WORLD_LOCAL_RANK",
                                    "MPI_LOCALRANKID", "PMI_LOCAL_RANK"},
                                   "MPI local rank"),
  };
  auto driver = megacu::adapters::mpi_cuda_nvshmem::make_driver(facts);
  return check_driver(driver, facts, nullptr);
}

int check_torch_env_contract() {
  megacu::adapters::torch_cuda_nvshmem::launch_facts facts{
      .rank = parse_required({"RANK"}, "Torch rank"),
      .world_size = parse_required({"WORLD_SIZE"}, "Torch world size"),
      .local_rank = parse_required({"LOCAL_RANK"}, "Torch local rank"),
  };
  auto driver = megacu::adapters::torch_cuda_nvshmem::make_driver(facts);
  return check_driver(driver, facts, nullptr);
}

} // namespace

int main() {
  auto *mode = std::getenv("MEGACU_ADAPTER_CONTRACT_MODE");
  if (mode == nullptr || std::string(mode) == "local") {
    return check_local_contract();
  }
  if (std::string(mode) == "mpi") {
    return check_mpi_env_contract();
  }
  if (std::string(mode) == "torch") {
    return check_torch_env_contract();
  }
  return fail(std::string("unknown mode: ") + mode);
}
