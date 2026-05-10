# GEMM Reduce-Scatter Megacu Tutorial

This tutorial explains how the GEMM Reduce-Scatter example uses Megacu to turn
small operator tasks into one CUDA mega-kernel.

```text
for each output tile:
  GEMM producer task computes partial tile
  sync-only task waits for that tile's EventTensor readiness
  reduce-scatter consumer task writes the rank-owned result
```

The example lives in `examples/cuda_nvshmem/gemm_reduce_scatter/`.

## What This Example Computes

Every rank computes:

```text
partial = A * B
```

Then reduce-scatter gives each rank ownership of only part of the result. In
this tiny example, row ownership is `row % n_pes`.

```text
if row belongs to this rank:
  out[row, col] = sum(partial[row, col] from every rank)
else:
  out[row, col] = 0
```

On one GPU, there is one rank, so every row belongs to the only rank and the
output equals normal GEMM.

On two GPUs, each rank receives only its owned rows. The two-rank test compares
Megacu output against both golden and baseline output buffers.

## Source Map

```text
common/gemm_reduce_scatter.h
  Public ABI for golden, baseline, host-orch, and seeded-orch calls.

common/gemm_reduce_scatter_runtime_recipe.cuh
  The Megacu recipe that submits per-tile producer/sync/consumer tasks.

megacu/gemm_reduce_scatter_megacu.cu
  CUDA operator bodies and linked Megacu runtime launch.

megacu/gemm_reduce_scatter_arena.cuh
  Host arena storage and host-orch arena construction.

golden/
  Reference implementation.

baseline/
  Handwritten CUDA+NVSHMEM comparison implementation.
```

## Megacu Features Used

### Raw CUDA-Like Arguments

The host code calls:

```cpp
cuda_nvshmem_gemm_reduce_scatter_host_orch(
    driver, a, b, partial, out, events, problem);
```

Pointers are raw. Megacu does not infer that `partial` is produced before
`out`. The recipe must explicitly submit tasks and dependencies.

### Per-Tile Operator Tasks

The recipe uses `dispatcher::single_tile(tile)`. That means each submitted task
has exactly one tile of work.

```cpp
auto gemm = orch.submit(gemm_tile_produce,
    attrs(single_tile(tile), event_tensor::notify(ready)));
```

This keeps synchronization at the operator-task level. If a program needs more
fine-grained synchronization, it should submit smaller tasks instead of asking
Megacu to split one task internally.

### Sync-Only Tasks

After the producer, the recipe submits:

```cpp
auto wait_ready = orch.sync(attrs(
    scheduler::depends_on(gemm),
    event_tensor::wait(ready, tile)));
```

This task has no operator kernel. It is scheduled after the GEMM task and
completes only when the EventTensor says all ranks have produced that tile.

### Reduce-Scatter Consumer

The consumer depends on the sync task:

```cpp
orch.submit(reduce_scatter_tile_consume,
    attrs(scheduler::depends_on(wait_ready),
          dispatcher::single_tile(tile)));
```

The consumer then sums local and remote partial values. Rows that this rank
does not own are written as zero.

## What The CUDA Mega-Kernel Does

`gemm_reduce_scatter_megacu.cu` links:

```text
runtime::device_persistent
  host-orch or seeded-orch execution model
  block_tile runtime loop
  explicit_asap scheduler
  tile_grid dispatcher
  CUDA+NVSHMEM EventTensor lowering
  GEMM and reduce-scatter operators
```

The runtime chooses ready tasks as soon as explicit dependencies and EventTensor
conditions allow them. Users do not program `scheduler.run`.

## 1 Host, 1 GPU Run

This validates golden, baseline, Megacu host-orch, and Megacu seeded-orch with
one CUDA device.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target megacu_cuda_gemm_reduce_scatter_correctness
ctest --test-dir build \
  -R '^cuda_gemm_reduce_scatter_correctness$' \
  --output-on-failure
```

Optional device selection:

```bash
MEGACU_TEST_CUDA_DEVICE=0 ctest --test-dir build \
  -R '^cuda_gemm_reduce_scatter_correctness$' \
  --output-on-failure
```

## 1 Host, 2 GPU Run

The two-rank run starts two ranks on one host:

```text
rank 0 -> GPU 0 -> local A values are 1
rank 1 -> GPU 1 -> local A values are 2
both ranks compute partial GEMM tiles
EventTensor waits for both ranks per tile
reduce-scatter sums the tile and keeps only owned rows
tests compare Megacu output to golden and baseline buffers
```

Manual local setup:

```bash
cmake -S . -B build-nvshmem -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DMEGACU_ENABLE_NVSHMEM_TESTS=ON

cmake --build build-nvshmem \
  --target megacu_nvshmem_two_rank_gemm_rs_smoke \
  --target megacu_mpi_two_rank_gemm_rs_smoke

ctest --test-dir build-nvshmem \
  -R 'nvshmem_two_rank_gemm_rs_correctness_megacu|mpi_two_rank_gemm_rs_correctness_megacu' \
  --output-on-failure
```

## Docker: Full Reproducible Run

```bash
export MEGACU_DOCKER_GPUS='"device=0,1"'
export MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8'
export MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem'

tools/cuda_nvshmem/run_two_card_docker.sh
```

## Docker: Focused GEMM-RS Run

```bash
docker build -f docker/cuda_nvshmem/Dockerfile \
  -t megacu-nvshmem:cuda12.8 .

docker run --rm \
  --gpus '"device=0,1"' \
  --ipc=host \
  -v "${PWD}:/workspace/megacu" \
  -w /workspace/megacu \
  megacu-nvshmem:cuda12.8 \
  bash -lc '
    rm -rf build-nvshmem &&
    cmake -S . -B build-nvshmem -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DMEGACU_ENABLE_NVSHMEM_TESTS=ON &&
    cmake --build build-nvshmem \
      --target megacu_cuda_gemm_reduce_scatter_correctness \
      --target megacu_nvshmem_two_rank_gemm_rs_smoke \
      --target megacu_mpi_two_rank_gemm_rs_smoke &&
    ctest --test-dir build-nvshmem \
      -R "cuda_gemm_reduce_scatter_correctness|nvshmem_two_rank_gemm_rs_correctness_megacu|mpi_two_rank_gemm_rs_correctness_megacu" \
      --output-on-failure
  '
```

## Torch UID Bootstrap Path

The Docker full gate also runs the Torch adapter. Torch is used only to
broadcast the NVSHMEM unique ID and launch two ranks; the Megacu recipe and
operators are unchanged.

Focused Torch run inside a configured Docker build:

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  --ipc=host \
  -v "${PWD}:/workspace/megacu" \
  -w /workspace/megacu \
  megacu-nvshmem:cuda12.8 \
  bash -lc '
    cmake --build build-nvshmem \
      --target megacu_adapter_contracts \
      --target megacu_torch_uid_two_rank_gemm_rs_smoke \
      --target megacu_torch_uid_two_rank_ag_gemm_smoke &&
    MEGACU_BUILD_DIR=build-nvshmem MEGACU_SKIP_BUILD=1 \
      OMP_NUM_THREADS=1 \
      torchrun --standalone --nproc_per_node=2 \
      tools/cuda_nvshmem/run_torch.py
  '
```

`run_torch.py` runs both distributed GEMM examples because it validates the
shared Torch UID-bootstrap adapter once and then launches GEMM-RS and AG-GEMM.

## How To Modify The Example

Change the task graph in `common/gemm_reduce_scatter_runtime_recipe.cuh`.
Change the per-tile math in `megacu/gemm_reduce_scatter_megacu.cu`.

When adding a new stage, add a slot, submit a task with explicit dependencies,
add the operator body, and compare against golden output in the tests.
