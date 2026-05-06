#pragma once

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

#if defined(MEGACU_RUNTIME_HOST_DEVICE)
#define MEGACU_TEST_CONTRACTS_HOST_DEVICE MEGACU_RUNTIME_HOST_DEVICE
#elif defined(__CUDACC__)
#define MEGACU_TEST_CONTRACTS_HOST_DEVICE __host__ __device__
#else
#define MEGACU_TEST_CONTRACTS_HOST_DEVICE
#endif

namespace megacu::runtime::contracts {

struct three_task_event_refs {
  event_tensor_ref ready = invalid_event_tensor_ref();
  task_ref producer = invalid_task_ref();
  task_ref wait_ready = invalid_task_ref();
  task_ref consumer = invalid_task_ref();
};

template <class Orch>
MEGACU_TEST_CONTRACTS_HOST_DEVICE three_task_event_refs
submit_three_task_event_recipe(Orch &orch) {
  three_task_event_refs refs;
  refs.ready = orch.event_tensor(
      attrs(event_tensor::shape(1, 1), event_tensor::wait_count(1)));
  refs.producer = orch.submit(
      operator_slot{3},
      attrs(dispatcher::tile_grid(1, 1), event_tensor::notify(refs.ready)));
  refs.wait_ready = orch.sync(attrs(scheduler::depends_on(refs.producer),
                                    event_tensor::wait(refs.ready)));
  refs.consumer = orch.submit(
      operator_slot{4}, attrs(scheduler::depends_on(refs.wait_ready),
                              dispatcher::tile_grid(1, 1)));
  return refs;
}

} // namespace megacu::runtime::contracts

#undef MEGACU_TEST_CONTRACTS_HOST_DEVICE
