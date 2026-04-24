# CUDA + NVSHMEM GEMM+AllReduce

This example demonstrates the first Megacu compiled orchestration lifecycle for
a CUDA platform and NVSHMEM backend. It provides two static targets:

- `cuda_nvshmem_gemm_allreduce_phased`: compute all local GEMM partials before
  reduction.
- `cuda_nvshmem_gemm_allreduce_overlap`: model co-resident compute and
  communication tasks so reduction may consume ready partial tiles.

## Files

- `gemm_allreduce.h`: public example descriptor, workspace views, target ABI,
  and program descriptions.
- `gemm_allreduce_phased_orchestrate.cc`: phased target entrypoint and linked
  metadata symbol.
- `gemm_allreduce_overlap_orchestrate.cc`: overlap target entrypoint and linked
  metadata symbol.
- `gemm_allreduce_orchestrate_common.h`: shared runtime validation and numeric
  path dispatch.
- `gemm_allreduce_kernels.cu`: CUDA numeric GEMM and reduced-output kernels.

## Execution

```text
rank 0 A0,B0 ─┐
              ├─ GEMM partial ─┐
rank 1 A1,B1 ─┘                │
                               ├─ NVSHMEM sum_reduce ── C on each rank
event: partial_ready ──────────┘
```

The current numeric path computes one small dense GEMM per rank, writes local
partials into symmetric scratch space, calls a backend-provided `sum_reduce_f32`
primitive, and writes the reduced matrix to `c`.

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

run target:
  validate workspace, events, launch, team, problem
  launch CUDA GEMM into partial
  call target ops sum_reduce_f32(reduced, partial)
  launch CUDA copy from reduced to c
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
CUDA numeric execution, and two-rank NVSHMEM correctness. It does not claim
performance, optimized tiling, or final persistent-kernel scheduling quality.
