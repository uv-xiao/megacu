#include <cassert>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>

#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm_runtime_recipe.cuh"
#include "examples/cuda_nvshmem/gemm_reduce_scatter/common/gemm_reduce_scatter_runtime_recipe.cuh"

namespace {

bool has_single_tile(megacu::runtime::attr_set const &attrs,
                     std::int64_t tile_id) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dispatch_single_tile &&
        attr.first == tile_id) {
      return true;
    }
  }
  return false;
}

bool has_wait_tile(megacu::runtime::attr_set const &attrs,
                   megacu::runtime::event_tensor_ref event,
                   std::int64_t tile_id) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_wait &&
        attr.event == event && attr.first == tile_id && attr.second == 1) {
      return true;
    }
  }
  return false;
}

bool has_notify(megacu::runtime::attr_set const &attrs,
                megacu::runtime::event_tensor_ref event) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_notify &&
        attr.event == event) {
      return true;
    }
  }
  return false;
}

void check_gemm_reduce_scatter_recipe() {
  int event_storage = 0;
  auto args = gemm_rs::runtime_args{
      .driver = {},
      .a = nullptr,
      .b = nullptr,
      .partial = nullptr,
      .out = nullptr,
      .events = &event_storage,
      .problem = {.m = 2, .n = 3, .k = 4, .tile_m = 1, .tile_n = 2}};
  args.driver.team.team_n_pes = 2;

  megacu::runtime::execution::host_orch::frame<16, 2, 16> frame;
  auto result = gemm_rs::build_runtime_recipe(frame, args);
  assert(result.code == megacu::status_code::ok);
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto deps = frame.deps();
  auto events = frame.event_tensors();
  assert(events.size() == 1);
  assert(tasks.size() == 12);
  assert(deps.size() == 8);

  for (std::int64_t tile = 0; tile < 4; ++tile) {
    auto const &producer = tasks[static_cast<std::size_t>(tile * 3)];
    auto const &sync = tasks[static_cast<std::size_t>(tile * 3 + 1)];
    auto const &consumer = tasks[static_cast<std::size_t>(tile * 3 + 2)];

    assert(producer.kind == megacu::runtime::task_kind::operator_body);
    assert(producer.op_slot == gemm_rs::runtime_slots::gemm_tile_produce);
    assert(producer.dep_count == 0);
    assert(has_single_tile(producer.attributes, tile));
    assert(has_notify(producer.attributes, events[0].ref));

    assert(sync.kind == megacu::runtime::task_kind::sync_only);
    assert(sync.dep_count == 1);
    assert(deps[sync.first_dep].value == static_cast<std::uint32_t>(tile * 3));
    assert(has_wait_tile(sync.attributes, events[0].ref, tile));

    assert(consumer.kind == megacu::runtime::task_kind::operator_body);
    assert(consumer.op_slot ==
           gemm_rs::runtime_slots::reduce_scatter_tile_consume);
    assert(consumer.dep_count == 1);
    assert(deps[consumer.first_dep].value ==
           static_cast<std::uint32_t>(tile * 3 + 1));
    assert(has_single_tile(consumer.attributes, tile));
  }
}

void check_allgather_gemm_recipe() {
  int event_storage = 0;
  auto args = ag_gemm::runtime_args{
      .driver = {},
      .a = nullptr,
      .b = nullptr,
      .gathered_b = nullptr,
      .out = nullptr,
      .events = &event_storage,
      .problem = {.m = 2, .n = 3, .k = 2, .tile_m = 1, .tile_n = 2, .tile_k = 1}};
  args.driver.team.team_n_pes = 2;

  megacu::runtime::execution::host_orch::frame<32, 2, 32> frame;
  auto result = ag_gemm::build_runtime_recipe(frame, args);
  assert(result.code == megacu::status_code::ok);
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto deps = frame.deps();
  auto events = frame.event_tensors();
  assert(events.size() == 1);
  assert(tasks.size() == 12);
  assert(deps.size() == 12);

  for (std::int64_t tile = 0; tile < 4; ++tile) {
    auto const &producer = tasks[static_cast<std::size_t>(tile)];
    assert(producer.kind == megacu::runtime::task_kind::operator_body);
    assert(producer.op_slot == ag_gemm::runtime_slots::allgather_tile_produce);
    assert(producer.dep_count == 0);
    assert(has_single_tile(producer.attributes, tile));
    assert(has_notify(producer.attributes, events[0].ref));
  }

  for (std::int64_t out_tile = 0; out_tile < 4; ++out_tile) {
    auto const sync_index = 4 + out_tile * 2;
    auto const consumer_index = sync_index + 1;
    auto const n_tile = out_tile % 2;
    auto const &sync = tasks[static_cast<std::size_t>(sync_index)];
    auto const &consumer = tasks[static_cast<std::size_t>(consumer_index)];

    assert(sync.kind == megacu::runtime::task_kind::sync_only);
    assert(sync.dep_count == 2);
    assert(deps[sync.first_dep].value ==
           static_cast<std::uint32_t>(n_tile));
    assert(deps[sync.first_dep + 1].value ==
           static_cast<std::uint32_t>(2 + n_tile));
    assert(has_wait_tile(sync.attributes, events[0].ref, n_tile));
    assert(has_wait_tile(sync.attributes, events[0].ref, 2 + n_tile));

    assert(consumer.kind == megacu::runtime::task_kind::operator_body);
    assert(consumer.op_slot == ag_gemm::runtime_slots::gemm_tile_consume);
    assert(consumer.dep_count == 1);
    assert(deps[consumer.first_dep].value ==
           static_cast<std::uint32_t>(sync_index));
    assert(has_single_tile(consumer.attributes, out_tile));
  }
}

} // namespace

int main() {
  check_gemm_reduce_scatter_recipe();
  check_allgather_gemm_recipe();
  return 0;
}
