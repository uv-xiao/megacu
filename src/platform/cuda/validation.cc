#include <megacu/detail/runtime_validation.h>

namespace megacu::detail {

megacu::status validate_cuda_launch(
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team) {
  if (launch.device_ordinal != team.cuda_device_ordinal) {
    return {megacu::status_code::invalid_argument, 3, "device mismatch"};
  }
  return {};
}

}  // namespace megacu::detail
