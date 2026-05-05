#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

#include <cstddef>

#include <megacu/runtime.h>

namespace {

struct host_driver {};

megacu::status tiny_decode_norm(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state->norm[index] = state->hidden[index] * 0.5F;
  }
  return {};
}

megacu::status tiny_decode_projection(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state->projection[index] =
        state->norm[index] + static_cast<float>(index + 1);
  }
  return {};
}

megacu::status tiny_decode_residual(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state->residual[index] = state->projection[index] + state->hidden[index];
  }
  return {};
}

megacu::status tiny_decode_mlp(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::mlp_size; ++index) {
    state->mlp[index] =
        state->residual[index] * state->residual[index] * 0.25F;
  }
  return {};
}

megacu::status tiny_decode_logits(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return {megacu::status_code::invalid_argument, 1,
            "missing tiny decode state"};
  }

  for (std::size_t vocab = 0;
       vocab < megacu::examples::tiny_decode::vocab_size; ++vocab) {
    float sum = 0.0F;
    for (std::size_t index = 0;
         index < megacu::examples::tiny_decode::mlp_size; ++index) {
      sum += state->mlp[index] *
             (static_cast<float>((index + 1) * (vocab + 1)) * 0.125F);
    }
    state->logits[vocab] = sum;
  }
  return {};
}

} // namespace

extern "C" int megacu_tiny_decode_megacu(
    megacu::examples::tiny_decode::buffers *state) {
  host_driver driver;
  auto phase = megacu::runtime::make_phase(
      driver, megacu::runtime::progress_model::asap);

  auto norm = phase.submit(
      megacu::runtime::op("tiny_decode_norm", &tiny_decode_norm), state);
  auto projection = phase.submit(
      megacu::runtime::op("tiny_decode_projection", &tiny_decode_projection),
      state,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(norm)));
  auto residual = phase.submit(
      megacu::runtime::op("tiny_decode_residual", &tiny_decode_residual), state,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(projection)));
  auto mlp = phase.submit(
      megacu::runtime::op("tiny_decode_mlp", &tiny_decode_mlp), state,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(residual)));
  phase.submit(
      megacu::runtime::op("tiny_decode_logits", &tiny_decode_logits), state,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(mlp)));

  auto status = phase.run();
  return status.code == megacu::status_code::ok ? 0 : 1;
}
