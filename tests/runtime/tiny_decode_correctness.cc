#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

int main() {
  megacu::examples::tiny_decode::buffers golden{};
  megacu::examples::tiny_decode::buffers baseline{};
  megacu::examples::tiny_decode::buffers megacu{};

  megacu::examples::tiny_decode::seed_inputs(golden);
  baseline = golden;
  megacu = golden;

  if (megacu_tiny_decode_golden(&golden) != 0) {
    return 1;
  }
  if (megacu_tiny_decode_baseline(&baseline) != 0) {
    return 2;
  }
  if (megacu_tiny_decode_megacu(&megacu) != 0) {
    return 3;
  }

  if (!megacu::examples::tiny_decode::nearly_equal(golden, baseline)) {
    return 4;
  }
  if (!megacu::examples::tiny_decode::nearly_equal(golden, megacu)) {
    return 5;
  }
  return 0;
}
