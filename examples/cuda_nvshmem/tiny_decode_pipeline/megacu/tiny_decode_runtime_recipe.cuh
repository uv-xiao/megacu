#pragma once

#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

#if defined(__CUDACC__)
#define TINY_DECODE_HOST_DEVICE __host__ __device__
#else
#define TINY_DECODE_HOST_DEVICE
#endif

namespace tiny_decode_runtime {

namespace tiny = megacu::examples::tiny_decode;

struct slots {
  static constexpr megacu::runtime::operator_slot norm{1};
  static constexpr megacu::runtime::operator_slot projection{2};
  static constexpr megacu::runtime::operator_slot residual{3};
  static constexpr megacu::runtime::operator_slot mlp{4};
  static constexpr megacu::runtime::operator_slot logits{5};
};

struct runtime_args {
  tiny::buffers *state = nullptr;
};

template <class Orch>
TINY_DECODE_HOST_DEVICE megacu::status build_recipe(Orch &orch,
                                                    runtime_args) {
  auto norm_ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(tiny::hidden_size, 1),
      megacu::runtime::event_tensor::wait_count(1)));
  auto norm = orch.submit(
      slots::norm,
      megacu::runtime::attrs(
          megacu::runtime::dispatcher::tile_grid(tiny::hidden_size, 1),
          megacu::runtime::event_tensor::notify(norm_ready)));
  auto norm_sync = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(norm),
      megacu::runtime::event_tensor::wait(norm_ready)));

  auto projection_ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(tiny::hidden_size, 1),
      megacu::runtime::event_tensor::wait_count(1)));
  auto projection = orch.submit(
      slots::projection,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(norm_sync),
          megacu::runtime::dispatcher::tile_grid(tiny::hidden_size, 1),
          megacu::runtime::event_tensor::notify(projection_ready)));
  auto projection_sync = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(projection),
      megacu::runtime::event_tensor::wait(projection_ready)));

  auto residual_ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(tiny::hidden_size, 1),
      megacu::runtime::event_tensor::wait_count(1)));
  auto residual = orch.submit(
      slots::residual,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(projection_sync),
          megacu::runtime::dispatcher::tile_grid(tiny::hidden_size, 1),
          megacu::runtime::event_tensor::notify(residual_ready)));
  auto residual_sync = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(residual),
      megacu::runtime::event_tensor::wait(residual_ready)));

  auto mlp_ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(tiny::mlp_size, 1),
      megacu::runtime::event_tensor::wait_count(1)));
  auto mlp = orch.submit(
      slots::mlp,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(residual_sync),
          megacu::runtime::dispatcher::tile_grid(tiny::mlp_size, 1),
          megacu::runtime::event_tensor::notify(mlp_ready)));
  auto mlp_sync = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(mlp),
      megacu::runtime::event_tensor::wait(mlp_ready)));

  (void)orch.submit(
      slots::logits,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(mlp_sync),
          megacu::runtime::dispatcher::tile_grid(tiny::vocab_size, 1)));

  return orch.current_status();
}

} // namespace tiny_decode_runtime

#undef TINY_DECODE_HOST_DEVICE
