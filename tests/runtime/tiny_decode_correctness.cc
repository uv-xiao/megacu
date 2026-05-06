#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

int main() {
  megacu::examples::tiny_decode::buffers golden{};
  megacu::examples::tiny_decode::buffers baseline{};
  megacu::examples::tiny_decode::buffers megacu_host_orch{};
  megacu::examples::tiny_decode::buffers megacu_seeded_orch{};
  megacu::examples::tiny_decode::buffers megacu_compat{};

  megacu::examples::tiny_decode::seed_inputs(golden);
  baseline = golden;
  megacu_host_orch = golden;
  megacu_seeded_orch = golden;
  megacu_compat = golden;

  if (megacu_tiny_decode_golden(&golden) != 0) {
    return 1;
  }
  if (megacu_tiny_decode_baseline(&baseline) != 0) {
    return 2;
  }
  if (megacu_tiny_decode_megacu_host_orch(&megacu_host_orch) != 0) {
    return 3;
  }
  if (megacu_tiny_decode_megacu_seeded_orch(&megacu_seeded_orch) != 0) {
    return 4;
  }
  if (megacu_tiny_decode_megacu(&megacu_compat) != 0) {
    return 5;
  }

  if (!megacu::examples::tiny_decode::nearly_equal(golden, baseline)) {
    return 6;
  }
  if (!megacu::examples::tiny_decode::nearly_equal(golden,
                                                    megacu_host_orch)) {
    return 7;
  }
  if (!megacu::examples::tiny_decode::nearly_equal(golden,
                                                    megacu_seeded_orch)) {
    return 8;
  }
  if (!megacu::examples::tiny_decode::nearly_equal(golden, megacu_compat)) {
    return 9;
  }
  return 0;
}
