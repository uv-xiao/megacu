#include <array>
#include <cassert>
#include <cstddef>

#include "examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce.h"

namespace {

gemm_ar_problem small_problem() {
  return {
      .m = 16,
      .n = 16,
      .k = 16,
      .tile_m = 16,
      .tile_n = 16};
}

}  // namespace

int main() {
  std::array<std::byte, 256> a{};
  std::array<std::byte, 256> b{};
  std::array<std::byte, 256> c{};
  std::array<std::byte, 256> partial{};
  std::array<std::byte, 256> scratch{};
  std::array<std::byte, 256> event_storage{};

  megacu::backend_id backend{1};
  megacu::session_id session{7};

  gemm_ar_workspace workspace{
      .a = {.data = a.data(), .bytes = static_cast<std::int64_t>(a.size())},
      .b = {.data = b.data(), .bytes = static_cast<std::int64_t>(b.size())},
      .partial = {
          .buffer = {
              .data = partial.data(),
              .bytes = static_cast<std::int64_t>(partial.size()),
              .backend = backend,
              .session = session}},
      .c = {.data = c.data(), .bytes = static_cast<std::int64_t>(c.size())},
      .scratch = scratch};

  megacu::event_storage_view events{
      .buffer = {
          .data = event_storage.data(),
          .bytes = static_cast<std::int64_t>(event_storage.size()),
          .backend = backend,
          .session = session}};

  megacu::cuda::launch_view launch{.stream = nullptr, .device_ordinal = 0};
  megacu::nvshmem::team_view team{
      .team = nullptr,
      .team_my_pe = 0,
      .team_n_pes = 2,
      .world_my_pe = 0,
      .world_n_pes = 2,
      .cuda_device_ordinal = 0,
      .backend = backend,
      .session = session};

  auto phased = cuda_nvshmem_gemm_allreduce_phased_orchestrate(
      workspace, events, launch, team, small_problem());
  assert(phased.code == megacu::status_code::ok);

  auto overlap = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, events, launch, team, small_problem());
  assert(overlap.code == megacu::status_code::ok);

  auto bad_team = team;
  bad_team.team_n_pes = 1;
  auto wrong_team_size = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, events, launch, bad_team, small_problem());
  assert(wrong_team_size.code == megacu::status_code::invalid_argument);

  auto bad_events = events;
  bad_events.buffer.session = megacu::session_id{999};
  auto wrong_session = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, bad_events, launch, team, small_problem());
  assert(wrong_session.code == megacu::status_code::invalid_argument);

  auto bad_problem = small_problem();
  bad_problem.tile_m = 0;
  auto wrong_problem = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      workspace, events, launch, team, bad_problem);
  assert(wrong_problem.code == megacu::status_code::invalid_argument);

  return 0;
}
