#include <cassert>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

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

} // namespace

int main() {
  megacu::runtime::execution::host_orch::frame<16, 2, 16> frame;
  int event_storage = 0;
  auto args = gemm_ar::runtime_args{
      .driver = {},
      .a = nullptr,
      .b = nullptr,
      .partial = nullptr,
      .out = nullptr,
      .events = &event_storage,
      .problem = {.m = 2, .n = 2, .k = 2, .tile_m = 1, .tile_n = 1}};
  args.driver.team.team_n_pes = 2;

  auto result = gemm_ar::build_runtime_recipe(frame, args);
  assert(result.code == megacu::status_code::ok);
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();

  assert(events.size() == 1);
  assert(tasks.size() == 12);
  assert(deps.size() == 8);
  assert(regions.size() == 1);
  assert(regions[0].sealed == 1);

  bool event_has_shape = false;
  bool event_has_wait_count = false;
  bool event_has_storage = false;
  bool event_has_scope = false;
  for (auto attr : events[0].attributes.entries()) {
    event_has_shape |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_shape &&
        attr.first == 2 && attr.second == 2;
    event_has_wait_count |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_wait_count &&
        attr.first == 2;
    event_has_storage |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_storage &&
        attr.pointer == &event_storage;
    event_has_scope |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_scope;
  }
  assert(event_has_shape && event_has_wait_count && event_has_storage &&
         event_has_scope);

  for (std::int64_t tile = 0; tile < 4; ++tile) {
    auto const producer_index = static_cast<std::size_t>(tile * 3);
    auto const sync_index = producer_index + 1;
    auto const consumer_index = producer_index + 2;
    auto const &producer = tasks[producer_index];
    auto const &sync = tasks[sync_index];
    auto const &consumer = tasks[consumer_index];

    assert(producer.kind == megacu::runtime::task_kind::operator_body);
    assert(producer.op_slot == gemm_ar::runtime_slots::gemm_tile_produce);
    assert(producer.dep_count == 0);
    assert(has_single_tile(producer.attributes, tile));
    assert(has_notify(producer.attributes, events[0].ref));

    assert(sync.kind == megacu::runtime::task_kind::sync_only);
    assert(sync.op_slot == megacu::runtime::invalid_operator_slot());
    assert(sync.dep_count == 1);
    assert(deps[sync.first_dep].value ==
           static_cast<std::uint32_t>(producer_index));
    assert(has_wait_tile(sync.attributes, events[0].ref, tile));

    assert(consumer.kind == megacu::runtime::task_kind::operator_body);
    assert(consumer.op_slot ==
           gemm_ar::runtime_slots::allreduce_tile_consume);
    assert(consumer.dep_count == 1);
    assert(deps[consumer.first_dep].value ==
           static_cast<std::uint32_t>(sync_index));
    assert(has_single_tile(consumer.attributes, tile));
  }
  return 0;
}
