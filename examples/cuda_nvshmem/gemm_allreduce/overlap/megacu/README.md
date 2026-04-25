# CUDA+NVSHMEM GEMM+AllReduce Overlap

This example owns the overlap Megacu target:

- target: `cuda_nvshmem_gemm_allreduce_overlap`
- program: `gemm_allreduce_overlap_program`
- scheduler mode: `co_resident_persistent`

The overlap target models the communication-progress case: a consumer may block
on tile readiness while producers are expected to keep making progress. Megacu
therefore requires co-resident compute and communication workers in the
materialized schedule metadata.

## Files

- `CMakeLists.txt`: example-owned orchestrate target.
- `gemm_allreduce_overlap_orchestrate.cc`: overlap target entrypoint and linked
  metadata symbol.
- `../../common/`: shared descriptors and runtime helper used by both phased
  and overlap variants.
- `../baseline/`: overlap pure CUDA/NVSHMEM baseline ownership point.
- `../../golden/`: local golden result generation and Megacu-free baseline
  entrypoints.

## Execution

```text
one persistent launch:
  comm workers:    wait partial_ready[tile] -> reduce tile -> next tile
  compute workers: produce partial tile     -> release event -> next tile

guard:
  scheduler metadata proves both worker roles are in the same residency group
```

Single-card execution uses the local fused persistent path. Two-card execution
uses CUDA+NVSHMEM device-side peer reads through the shared validation tests.

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_overlap
MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build -R 'component_metadata_contracts|cuda_gemm_allreduce_correctness' --output-on-failure
```

For two-card Docker validation:

```sh
tools/cuda_nvshmem/run_two_card_docker.sh
```

## Scope

This example proves the overlap target ABI, the co-resident progress guard, and
numeric correctness through the shared CUDA/NVSHMEM validation tests. It does
not yet claim a performance-tuned multi-card fused overlap implementation.
