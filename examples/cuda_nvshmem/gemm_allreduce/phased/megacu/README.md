# CUDA+NVSHMEM GEMM+AllReduce Phased

This example owns the phased Megacu target:

- target: `cuda_nvshmem_gemm_allreduce_phased`
- progress: `asap`
- host call shape: `driver + raw target arguments`

The orchestrate entrypoint constructs one Megacu runtime phase from small
operator submissions. GEMM is submitted first and notifies EventTensor
readiness, a sync-only task is submitted second and waits on that EventTensor,
and AllReduce depends on that sync task. The target runtime
validates the submitted graph and lowers this PR #4 proof to one native phased
Megacu launcher instead of launching each submitted operator from the host.
That launcher links the common runtime-strategy pieces: the explicit ASAP scheduler
under `include/megacu/scheduler/`, the tile-grid dispatcher under
`include/megacu/dispatcher/`, the generic CUDA mega-kernel shell under
`include/megacu/platform/cuda/`, and the NVSHMEM event tensor handler under
`include/megacu/backends/nvshmem/`. The handwritten monolithic mega-kernel
baseline lives in `../baseline/`, not here. Operator bodies do not program
readiness waits or signals, and readiness is not a contract based on CUDA
stream ordering.

## Files

- `CMakeLists.txt`: links the phased orchestrate target to runtime components.
- `gemm_allreduce_phased_orchestrate.cc`: host-visible orchestrate entrypoint.
- `../../common/`: shared raw-argument contract and runtime helper.
- `../baseline/`: phased pure CUDA/NVSHMEM baseline ownership point.
- `../../golden/`: local golden result generation and Megacu-free baseline
  entrypoints.

## Usage

Build the phased Megacu target from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_phased
```

Build the runnable single-process CUDA checks:

```sh
cmake --build build --target \
  megacu_direct_orchestrate_smoke \
  megacu_cuda_orchestrate_smoke \
  megacu_cuda_multicard_smoke \
  megacu_cuda_gemm_allreduce_correctness
```

Run the compiled 1-host-1-GPU correctness binary directly:

```sh
MEGACU_TEST_CUDA_DEVICE=0 \
  ./build/examples/cuda_nvshmem/gemm_allreduce/megacu_cuda_gemm_allreduce_correctness
```

Run the compiled local two-device smoke binary directly. This checks that the
same host process can construct two CUDA driver views and call the phased
Megacu orchestrate on each selected GPU. It does not initialize NVSHMEM:

```sh
MEGACU_TEST_CUDA_DEVICES=0,1 \
  ./build/examples/cuda_nvshmem/gemm_allreduce/megacu_cuda_multicard_smoke
```

The equivalent focused CTest command is:

```sh
ctest --test-dir build -R 'runtime_linked_surface_contract|direct_orchestrate_smoke|cuda_gemm_allreduce_correctness' --output-on-failure
```

## Two-Rank CUDA+NVSHMEM Run

The distributed CUDA+NVSHMEM binary is built only when NVSHMEM tests are
enabled. The build must be able to find:

- CUDA with `nvcc`;
- `nvshmem.h`;
- `libnvshmem_host`;
- `libnvshmem_device`;
- `nvshmrun` or `nvshmemrun`.

The repository Docker helper sets up that environment and runs the Megacu
two-rank test:

```sh
MEGACU_DOCKER_GPUS='"device=0,1"' \
  tools/cuda_nvshmem/run_two_card_docker.sh
```

For a native host install, set the NVSHMEM paths before configuring CMake. The
exact install prefix may differ, but the binary needs the NVSHMEM launcher on
`PATH` and the host library on `LD_LIBRARY_PATH`:

```sh
export NVSHMEM_HOME=/opt/nvidia/nvshmem
export PATH="$NVSHMEM_HOME/bin:/usr/bin/nvshmem_12:$PATH"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"

cmake -S . -B build-nvshmem -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DMEGACU_ENABLE_NVSHMEM_TESTS=ON
cmake --build build-nvshmem --target megacu_nvshmem_two_rank_smoke
```

Run the compiled two-rank Megacu binary with NVSHMEM:

```sh
nvshmrun -np 2 \
  ./build-nvshmem/examples/cuda_nvshmem/gemm_allreduce/megacu_nvshmem_two_rank_smoke \
  megacu_phased
```

Run the matching CTest:

```sh
ctest --test-dir build-nvshmem \
  -R 'nvshmem_two_rank_gemm_ar_correctness_megacu' \
  --output-on-failure
```

The `megacu_phased` argument selects the Megacu path inside the smoke binary.
This Megacu README uses that mode as the expected run path; handwritten
comparison details live outside this directory.

## Scope

This example proves the phased runtime-linked target ABI and numeric
correctness. It does not infer dependencies from pointers; missing dependency
attributes are the target author's responsibility. It is intentionally tiny:
the linked CUDA path composes GEMM and AllReduce operator task bodies with
EventTensor notify/wait handlers into one mega-kernel for the authored
GEMM -> EventTensor sync-only readiness -> AllReduce task graph. The same
code path targets both 1-host/1-GPU and 1-host/2-GPU; only the backend
lowering changes how EventTensor wait reads peer readiness and data.
