#include <megacu/platform/cuda.h>

namespace megacu::cuda {

bool has_stream(launch_view launch) noexcept {
  return launch.stream != nullptr;
}

}  // namespace megacu::cuda

extern "C" int megacu_platform_cuda_component() {
  return 1;
}
