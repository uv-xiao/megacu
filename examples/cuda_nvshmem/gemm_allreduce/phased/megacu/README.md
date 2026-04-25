# CUDA+NVSHMEM GEMM+AllReduce Phased

This example owns the phased Megacu target:

- target: `cuda_nvshmem_gemm_allreduce_phased`
- program: `gemm_allreduce_phased_program`
- scheduler mode: `phased`

The phased target models MPK-style correctness first: GEMM tile producers
release tile-readiness events, and reduction consumers acquire those events
without requiring compute and communication workers to be co-resident in one
persistent launch.

## Files

- `CMakeLists.txt`: example-owned orchestrate target.
- `gemm_allreduce_phased_orchestrate.cc`: phased target entrypoint and linked
  metadata symbol.
- `../../common/`: shared descriptors and runtime helper used by both phased
  and overlap variants.
- `../baseline/`: phased pure CUDA/NVSHMEM baseline ownership point.
- `../../golden/`: local golden result generation and Megacu-free baseline
  entrypoints.

## Execution

```text
GEMM tile producer     partial_ready[tile, rank]      reduction consumer
       │                         │                            │
       ├─ writes partial tile ───▶│                            │
       ├─ releases event ────────▶│                            │
                                 └─ acquire when ready ───────▶│
                                                              └─ reduce tile
```

The same authored program supports `team_n_pes == 1` and `team_n_pes == 2`.
The dispatcher/backend capability envelope chooses local CUDA behavior or
CUDA+NVSHMEM behavior at runtime.

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_phased
MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build -R 'direct_orchestrate_smoke|cuda_orchestrate_smoke|cuda_gemm_allreduce_correctness' --output-on-failure
```

For two-card Docker validation:

```sh
tools/cuda_nvshmem/run_two_card_docker.sh
```

## Scope

This example proves the phased target ABI, materialized phased schedule
metadata, and numeric correctness through the shared CUDA/NVSHMEM validation
tests. It does not claim optimized tiling, arbitrary PE counts, or dynamic
scheduling.
