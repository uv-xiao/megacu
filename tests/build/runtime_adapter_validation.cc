#include <cassert>
#include <cstddef>
#include <cstdint>

#include <megacu/detail/runtime_validation.h>
#include <megacu/views.h>

namespace {

megacu::nvshmem::team_view team() {
  return {
      .team = nullptr,
      .team_my_pe = 0,
      .team_n_pes = 2,
      .world_my_pe = 0,
      .world_n_pes = 2,
      .cuda_device_ordinal = 6,
      .backend = {.value = 7},
      .session = {.value = 11}};
}

}  // namespace

int main() {
  auto valid_team = team();
  auto launch = megacu::cuda::launch_view{
      .stream = reinterpret_cast<void *>(0x1000),
      .device_ordinal = 6};

  assert(megacu::detail::validate_cuda_launch(launch, valid_team).code ==
         megacu::status_code::ok);

  auto wrong_device = launch;
  wrong_device.device_ordinal = 5;
  assert(megacu::detail::validate_cuda_launch(wrong_device, valid_team).code ==
         megacu::status_code::invalid_argument);

  assert(megacu::detail::validate_nvshmem_team(valid_team, 2).code ==
         megacu::status_code::ok);

  auto bad_team = valid_team;
  bad_team.team_n_pes = 3;
  assert(megacu::detail::validate_nvshmem_team(bad_team, 2).code ==
         megacu::status_code::invalid_argument);

  assert(megacu::detail::validate_nvshmem_team(valid_team, 1).code ==
         megacu::status_code::invalid_argument);

  std::byte event_storage[32]{};
  float partial_storage[16]{};
  megacu::event_storage_view events{
      .buffer = {
          .data = event_storage,
          .bytes = static_cast<std::int64_t>(sizeof(event_storage)),
          .backend = valid_team.backend,
          .session = valid_team.session}};
  megacu::symmetric_tensor_view partial{
      .buffer = {
          .data = partial_storage,
          .bytes = static_cast<std::int64_t>(sizeof(partial_storage)),
          .backend = valid_team.backend,
          .session = valid_team.session},
      .type = megacu::dtype::f32};

  assert(megacu::detail::validate_nvshmem_symmetric_storage(
             events, partial, valid_team, 16)
             .code == megacu::status_code::ok);

  events.buffer.session.value = 12;
  assert(megacu::detail::validate_nvshmem_symmetric_storage(
             events, partial, valid_team, 16)
             .code == megacu::status_code::invalid_argument);

  return 0;
}
