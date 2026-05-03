# CUDA+NVSHMEM GEMM+AllReduce

This example family validates the PR #4 Megacu CUDA+NVSHMEM path with a tiny
phased GEMM+AllReduce target.

## Layout

```text
common/             shared direct target signature and runtime helper
golden/             Megacu-free golden and baseline entrypoints
phased/baseline/    phased pure CUDA/NVSHMEM baseline ownership point
phased/megacu/      phased Megacu target
```

## Execution

```text
host calls cuda_nvshmem_gemm_allreduce_phased_orchestrate(driver, raw args...)
        |
        v
orchestrate submits GEMM and AllReduce operators with raw args
        |
        v
sync-only readiness task carries scheduler::depends_on(gemm)
        |
        v
AllReduce carries scheduler::depends_on(sync)
        |
        v
linked CUDA/NVSHMEM phased native path runs one fused Megacu launcher
```

The active Megacu proof is phased only. It covers single-GPU CUDA execution and
two-GPU device-side NVSHMEM execution when the NVSHMEM test option is enabled.

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_phased
ctest --test-dir build -R 'cuda_gemm_allreduce_correctness' --output-on-failure
```

Two-GPU NVSHMEM validation requires `MEGACU_ENABLE_NVSHMEM_TESTS=ON` and a
working `nvshmrun` installation:

```sh
cmake -S . -B build-nvshmem -DMEGACU_ENABLE_NVSHMEM_TESTS=ON
cmake --build build-nvshmem --target megacu_nvshmem_two_rank_smoke
ctest --test-dir build-nvshmem -R 'nvshmem_two_rank_gemm_ar_correctness_(golden|megacu)' --output-on-failure
```
