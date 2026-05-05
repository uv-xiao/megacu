#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

#include <cstddef>

namespace {

void compute_norm(megacu::examples::tiny_decode::buffers &state) {
  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state.norm[index] = state.hidden[index] * 0.5F;
  }
}

void compute_projection(megacu::examples::tiny_decode::buffers &state) {
  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state.projection[index] = state.norm[index] + static_cast<float>(index + 1);
  }
}

void compute_residual(megacu::examples::tiny_decode::buffers &state) {
  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::hidden_size; ++index) {
    state.residual[index] = state.projection[index] + state.hidden[index];
  }
}

void compute_mlp(megacu::examples::tiny_decode::buffers &state) {
  for (std::size_t index = 0;
       index < megacu::examples::tiny_decode::mlp_size; ++index) {
    state.mlp[index] = state.residual[index] * state.residual[index] * 0.25F;
  }
}

void compute_logits(megacu::examples::tiny_decode::buffers &state) {
  for (std::size_t vocab = 0;
       vocab < megacu::examples::tiny_decode::vocab_size; ++vocab) {
    float sum = 0.0F;
    for (std::size_t index = 0;
         index < megacu::examples::tiny_decode::mlp_size; ++index) {
      sum += state.mlp[index] *
             (static_cast<float>((index + 1) * (vocab + 1)) * 0.125F);
    }
    state.logits[vocab] = sum;
  }
}

} // namespace

extern "C" int megacu_tiny_decode_baseline(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return 1;
  }

  compute_norm(*state);
  compute_projection(*state);
  compute_residual(*state);
  compute_mlp(*state);
  compute_logits(*state);
  return 0;
}
