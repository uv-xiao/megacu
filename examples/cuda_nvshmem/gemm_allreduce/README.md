# CUDA+NVSHMEM GEMM+AllReduce

This example family validates Megacu's CUDA+NVSHMEM path with one logical
GEMM+AllReduce design and two scheduling variants.

## Layout

```text
common/             shared descriptors and runtime helpers
golden/             Megacu-free golden and baseline entrypoints
phased/baseline/    phased pure CUDA/NVSHMEM baseline ownership point
phased/megacu/      phased Megacu target
overlap/baseline/   overlap pure CUDA/NVSHMEM baseline ownership point
overlap/megacu/     co-resident persistent Megacu target
```

## Execution

```text
GEMM tiles produce partial C
        |
        v
tile-ready events release communication work
        |
        v
single-card local reduction or two-card device-side NVSHMEM reduction
```

The phased Megacu variant allows reduction work to consume ready tiles without
requiring a co-resident persistent launch. The overlap Megacu variant uses a
persistent schedule with compute and communication workers in one residency
group so blocking communication waits have a progress guard.

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_phased
cmake --build build --target cuda_nvshmem_gemm_allreduce_overlap
MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build -R 'cuda_gemm_allreduce_correctness|nvshmem_two_rank_smoke' --output-on-failure
```

Two-card Docker validation uses the shared platform/backend script:

```sh
tools/cuda_nvshmem/run_two_card_docker.sh
```
