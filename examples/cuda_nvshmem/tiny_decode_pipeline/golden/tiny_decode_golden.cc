#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

#include <cmath>
#include <cstddef>

namespace megacu::examples::tiny_decode {
namespace {

void compute_norm(buffers &state) {
  for (std::size_t index = 0; index < hidden_size; ++index) {
    state.norm[index] = state.hidden[index] * 0.5F;
  }
}

void compute_projection(buffers &state) {
  for (std::size_t index = 0; index < hidden_size; ++index) {
    state.projection[index] =
        state.norm[index] + static_cast<float>(index + 1);
  }
}

void compute_residual(buffers &state) {
  for (std::size_t index = 0; index < hidden_size; ++index) {
    state.residual[index] = state.projection[index] + state.hidden[index];
  }
}

void compute_mlp(buffers &state) {
  for (std::size_t index = 0; index < mlp_size; ++index) {
    state.mlp[index] =
        state.residual[index] * state.residual[index] * 0.25F;
  }
}

void compute_logits(buffers &state) {
  for (std::size_t vocab = 0; vocab < vocab_size; ++vocab) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < mlp_size; ++index) {
      sum += state.mlp[index] *
             (static_cast<float>((index + 1) * (vocab + 1)) * 0.125F);
    }
    state.logits[vocab] = sum;
  }
}

bool equal_array(float const *lhs, float const *rhs, std::size_t size) {
  constexpr float tolerance = 1.0e-5F;
  for (std::size_t index = 0; index < size; ++index) {
    if (std::fabs(lhs[index] - rhs[index]) > tolerance) {
      return false;
    }
  }
  return true;
}

} // namespace

void seed_inputs(buffers &state) {
  for (std::size_t index = 0; index < hidden_size; ++index) {
    state.hidden[index] = static_cast<float>(index + 1);
    state.norm[index] = 0.0F;
    state.projection[index] = 0.0F;
    state.residual[index] = 0.0F;
  }
  for (std::size_t index = 0; index < mlp_size; ++index) {
    state.mlp[index] = 0.0F;
  }
  for (std::size_t index = 0; index < vocab_size; ++index) {
    state.logits[index] = 0.0F;
  }
}

bool nearly_equal(buffers const &lhs, buffers const &rhs) {
  return equal_array(lhs.hidden, rhs.hidden, hidden_size) &&
         equal_array(lhs.norm, rhs.norm, hidden_size) &&
         equal_array(lhs.projection, rhs.projection, hidden_size) &&
         equal_array(lhs.residual, rhs.residual, hidden_size) &&
         equal_array(lhs.mlp, rhs.mlp, mlp_size) &&
         equal_array(lhs.logits, rhs.logits, vocab_size);
}

} // namespace megacu::examples::tiny_decode

extern "C" int megacu_tiny_decode_golden(
    megacu::examples::tiny_decode::buffers *state) {
  if (state == nullptr) {
    return 1;
  }

  megacu::examples::tiny_decode::compute_norm(*state);
  megacu::examples::tiny_decode::compute_projection(*state);
  megacu::examples::tiny_decode::compute_residual(*state);
  megacu::examples::tiny_decode::compute_mlp(*state);
  megacu::examples::tiny_decode::compute_logits(*state);
  return 0;
}
