# CUDA + NVSHMEM GEMM+AllReduce

This example demonstrates the first Megacu compiled orchestration lifecycle for
a CUDA platform and NVSHMEM backend. It has four golden native baselines and two
Megacu implementations:

- `golden_phased_single_card`: pure CUDA, two-kernel tiled GEMM producer plus
  tiled consumer.
- `golden_phased_multi_card`: CUDA tiled GEMM producer plus device-side
  NVSHMEM peer reads in a persistent tiled allreduce consumer.
- `golden_overlap_single_card`: pure CUDA fused persistent kernel with reserved
  communication CTAs and compute CTAs resident together.
- `golden_overlap_multi_card`: CUDA/NVSHMEM device-side tiled allreduce with
  the same multi-card semantics as the phased baseline.
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
  rank-local tiles ── symmetric partial buffer ── device NVSHMEM peer reads ── C
```

The two-rank Docker path installs NVIDIA's complete NVSHMEM package so the
example links both `libnvshmem_host` and `libnvshmem_device`. Host NVSHMEM is
used only for process bootstrap, symmetric allocation, and collective entry/exit
guards. Data movement for the golden multi-card paths and the Megacu multi-card
paths is issued from CUDA device code with NVSHMEM device APIs.

## Megacu Config Capability

The example uses one Megacu component config:

| Field | Value |
| --- | --- |
| `NAME` | `cuda_nvshmem_static` |
| `PLATFORM` | `cuda` |
| `BACKEND` | `nvshmem` |
| `DISPATCHER` | `tiled_compute_comm_dispatch` |
| `SCHEDULER` | `static_persistent` |
| `KERNEL_LOWERING` | `persistent_stitch` |

Capability range:

- One design covers single-card and two-card runs. The runtime `team_view`
  selects `team_n_pes == 1` for local CUDA or `team_n_pes == 2` for CUDA plus
  NVSHMEM over symmetric partial/event storage.
- Supported numeric shape is contiguous `f32` GEMM+AllReduce with caller-owned
  CUDA stream, CUDA device ordinal, symmetric partial buffer, and symmetric
  event storage from one NVSHMEM session.
- Multi-card communication is two PE, single host, CUDA+NVSHMEM, and uses
  device-side peer reads after per-tile readiness. The host side only brackets
  collective lifetime so event storage is not recycled while a peer is polling.
- This config does not claim arbitrary PE counts, non-contiguous layouts,
  non-`f32` types, dynamic scheduling, MPI/Torch distributed integration, or
  performance-tuned fused multi-card overlap. Those require separate backend
  capability entries rather than changing the program design.

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
  single-card:
    launch one persistent fused kernel with comm CTAs and compute CTAs
  multi-card:
    launch persistent GEMM producer and persistent NVSHMEM allreduce consumer
    consumer waits for local and peer tile readiness before reducing

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
two-rank device-side NVSHMEM correctness. It does not claim performance,
optimized tiling, arbitrary process launchers, or final fused multi-card
overlap quality.
