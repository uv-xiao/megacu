# CUDA + NVSHMEM GEMM+AllReduce

This example demonstrates the first Megacu compiled orchestration lifecycle for
a CUDA platform and NVSHMEM backend. It has four golden native baselines and two
Megacu implementations:

- `golden_phased_single_card`: pure CUDA, two-kernel tiled GEMM producer plus
  tiled consumer.
- `golden_phased_multi_card`: CUDA local tiled GEMM plus NVSHMEM host
  sum-reduce for two ranks.
- `golden_overlap_single_card`: pure CUDA fused persistent kernel with reserved
  communication CTAs and compute CTAs resident together.
- `golden_overlap_multi_card`: fused CUDA local persistent work plus NVSHMEM
  host sum-reduce for two ranks.
- `cuda_nvshmem_gemm_allreduce_phased`: Megacu phased target using tile-ready
  dependencies and the same native golden semantics.
- `cuda_nvshmem_gemm_allreduce_overlap`: Megacu overlap target using the fused
  persistent native overlap semantics.

## Files

- `gemm_allreduce.h`: public example descriptor, workspace views, target ABI,
  and program descriptions.
- `gemm_allreduce_phased_orchestrate.cc`: phased target entrypoint and linked
  metadata symbol.
- `gemm_allreduce_overlap_orchestrate.cc`: overlap target entrypoint and linked
  metadata symbol.
- `gemm_allreduce_orchestrate_common.h`: shared runtime validation and numeric
  path dispatch.
- `golden/`: Megacu-free CUDA/NVSHMEM golden baseline API and kernels.
- `megacu/`: two Megacu native wrapper implementations, phased and overlap.
- `CMakeLists.txt`: all example targets and example-specific tests.

## Execution

```text
phased:
  compute stream: GEMM tile 0 ── release ready[0] ── GEMM tile 1 ── ...
  comm stream:       wait ready[0] ── consume tile 0 ── wait ready[1] ── ...

overlap:
  CTA 0: comm loop waits ready[tile] and consumes tiles
  CTA 1..N: persistent GEMM loops produce tiles and release ready[tile]

multi-card:
  rank-local tiles ── symmetric partial buffer ── NVSHMEM sum_reduce ── C
```

The two-rank Docker path uses the NVSHMEM host collective exported by the
packaged CUDA 12.8 NVSHMEM library. Device-side NVSHMEM reduction inside the
persistent overlap kernel remains a backend upgrade once the device archive is
available in the validation image.

## Pseudocode

```cpp
describe program:
  workspace = resource<gemm_ar_workspace>()
  tile = domain(m_tiles, n_tiles)
  rank = participant<rank_lane>()
  partial_ready = event(tile, rank)

  submit gemm_tile_produce over tile on compute:
    args a, b, partial
    release partial_ready

  submit allreduce_tile_consume over tile on reduce:
    args partial, c
    acquire partial_ready

golden phased:
  clear tile readiness
  launch persistent GEMM producer on compute stream
  launch tiled consumer on comm stream
  consumer waits for each tile readiness before copying/reducing

golden overlap:
  launch one persistent fused kernel
  if CTA is a comm CTA:
    for assigned tiles:
      wait tile readiness
      consume tile
  else:
    for assigned tiles:
      compute GEMM tile
      release tile readiness

megacu target:
  validate workspace, events, launch, team, problem
  select single-card or multi-card path from team envelope
  call the phased or overlap native implementation linked by CMake
```

## Usage

Build and run the host CUDA tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
MEGACU_TEST_CUDA_DEVICES=5,6 MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build --output-on-failure
```

Run the Docker-provisioned two-rank NVSHMEM validation:

```sh
tools/cuda_nvshmem/gemm_allreduce/run_two_card_docker.sh
```

The Docker script expects Docker with NVIDIA GPU runtime support and defaults
to GPUs `5,6`. Override with:

```sh
MEGACU_DOCKER_GPUS='"device=0,1"' \
  tools/cuda_nvshmem/gemm_allreduce/run_two_card_docker.sh
```

## Scope

This example proves descriptor authoring, linked metadata, direct target ABI,
CUDA golden correctness, Megacu correctness, example-local CMake ownership, and
two-rank NVSHMEM correctness. It does not claim performance, optimized tiling,
or final device-side NVSHMEM persistent communication quality.
