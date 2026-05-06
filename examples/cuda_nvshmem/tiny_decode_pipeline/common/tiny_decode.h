#pragma once

#include <cstddef>

namespace megacu::examples::tiny_decode {

inline constexpr std::size_t hidden_size = 4;
inline constexpr std::size_t mlp_size = 4;
inline constexpr std::size_t vocab_size = 3;

struct buffers {
  float hidden[hidden_size];
  float norm[hidden_size];
  float projection[hidden_size];
  float residual[hidden_size];
  float mlp[mlp_size];
  float logits[vocab_size];
};

void seed_inputs(buffers &state);
bool nearly_equal(buffers const &lhs, buffers const &rhs);

} // namespace megacu::examples::tiny_decode

extern "C" int megacu_tiny_decode_golden(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_baseline(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_megacu_host_orch(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_megacu_seeded_orch(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_megacu(
    megacu::examples::tiny_decode::buffers *state);
