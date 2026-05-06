# CUDA+NVSHMEM GEMM-AllReduce

This example family validates the CUDA+NVSHMEM runtime path with a tiny
GEMM-AllReduce target. The Megacu variant is composed from operator tasks and
sync-only tasks. The handwritten mega-kernel is a baseline only.

## Layout

```text
common/             shared target ABI and Megacu recipe
golden/             Megacu-free golden entrypoints
phased/baseline/    handwritten CUDA/NVSHMEM manual mega-kernel baseline
megacu/             host-orch and seeded-orch Megacu runtime variant
```

## Execution

```text
orchestrate(driver, a, b, partial, out, events, problem)
  -> build_runtime_recipe(...)
  -> EventTensor ready object
  -> GEMM tile operator task
  -> sync-only wait task
  -> AllReduce tile operator task
  -> runtime-owned mega-kernel loop
```

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_megacu
ctest --test-dir build -R 'cuda_gemm_allreduce_correctness|runtime_gemm_allreduce' --output-on-failure
```

Two-GPU NVSHMEM validation requires `MEGACU_ENABLE_NVSHMEM_TESTS=ON` and a
working `nvshmrun` installation:

```sh
cmake -S . -B build-nvshmem -DMEGACU_ENABLE_NVSHMEM_TESTS=ON
cmake --build build-nvshmem --target megacu_nvshmem_two_rank_smoke
ctest --test-dir build-nvshmem -R 'nvshmem_two_rank_gemm_ar_correctness_(golden|manual_baseline|megacu)' --output-on-failure
```
