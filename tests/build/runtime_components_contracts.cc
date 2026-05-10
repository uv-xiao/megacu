#include <cassert>
#include <cstdint>

#include <megacu/runtime.h>

extern "C" int megacu_platform_cuda_component();
extern "C" int megacu_backend_nvshmem_component();

namespace {

struct driver {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
};

} // namespace

int main() {
  assert(megacu_platform_cuda_component() == 1);
  assert(megacu_backend_nvshmem_component() == 1);

  driver runtime_driver{};
  assert(!megacu::cuda::has_stream(runtime_driver.launch));
  runtime_driver.launch.stream = &runtime_driver;
  assert(megacu::cuda::has_stream(runtime_driver.launch));

  assert(megacu::nvshmem::team_size(runtime_driver.team) == 1);
  assert(!megacu::nvshmem::has_remote_pes(runtime_driver.team));
  runtime_driver.team.team_n_pes = 2;
  assert(megacu::nvshmem::team_size(runtime_driver.team) == 2);
  assert(megacu::nvshmem::has_remote_pes(runtime_driver.team));

  int ready[1]{};
  megacu::cuda::device::signal_ready(ready, 0, 7);
  assert(megacu::cuda::device::load_ready(ready, 0) == 7);
  assert(megacu::nvshmem::device::remote_int(ready, 0, 0) == 7);
  return 0;
}
