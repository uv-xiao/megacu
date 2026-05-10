# AllGather GEMM Megacu Tutorial

This tutorial explains the AllGather GEMM example. It is the most useful
example for understanding larger fan-in patterns because it uses both
`depends_on_many` and `event_tensor::wait_strided`.

The example lives in `examples/cuda_nvshmem/allgather_gemm/`.

## What This Example Computes

The GEMM is:

```text
out = A * B
```

but `B` is distributed by K rows across ranks. Before a GEMM output tile can
run, the needed rows of `B` must be gathered from the owning ranks into
`gathered_b`.

```text
all-gather producer tasks:
  fill gathered_b[k, n] from local or remote rank storage

GEMM consumer tasks:
  use gathered_b to compute output tiles
```

On one GPU, all of `B` is local. On two GPUs, each rank owns part of K, and
the all-gather producer reads local or remote pieces with NVSHMEM.

## Source Map

```text
common/allgather_gemm.h
  Public ABI and problem shape.

common/allgather_gemm_runtime_recipe.cuh
  Megacu recipe with dependency groups and strided EventTensor waits.

megacu/allgather_gemm_megacu.cu
  All-gather operator, GEMM operator, operator dispatch, and runtime launch.

megacu/allgather_gemm_arena.cuh
  Host arena storage for host-orch.

golden/
  Reference implementation.

baseline/
  Handwritten CUDA+NVSHMEM comparison implementation.
```

## Megacu Features Used

### Dependency Groups

For one output N tile, the recipe may need many K tiles of gathered `B`.
Instead of consuming one inline attr per producer dependency, it creates an
orch-owned dependency group:

```cpp
auto gathered_deps = orch.dependency_group();

for (k_tile in K tiles) {
  auto producer = orch.submit(allgather_tile_produce, attrs(...));
  gathered_deps.push(producer);
}
```

The scheduler later sees:

```cpp
scheduler::depends_on_many(gathered_deps.ref())
```

This is scheduler readiness state. It says all producer tasks in the group must
finish before the sync task can be considered ready.

### Strided EventTensor Wait

The same sync task also waits for EventTensor state:

```cpp
event_tensor::wait_strided(gathered, n_tile, k_tiles, n_tiles)
```

This means:

```text
wait for gathered event entries:
  n_tile
  n_tile + n_tiles
  n_tile + 2 * n_tiles
  ...
```

Those are the gathered K tiles needed for this N tile. This EventTensor wait is
not a scheduler dependency. It is the sync-only task's completion condition.

### Operator Granularity

The all-gather producer task uses `single_tile(gathered_tile)`, where
`gathered_tile = k_tile * n_tiles + n_tile`.

The GEMM consumer task uses `single_tile(output_tile)`, where
`output_tile = m_tile * n_tiles + n_tile`.

If you want more parallelism or more synchronization points, submit more
operator tasks. Megacu does not split one submitted task into hidden subtasks.

## What The CUDA Mega-Kernel Does

The runtime links:

```text
host-orch or seeded-orch execution model
block_tile loop
explicit_asap scheduler
tile_grid dispatcher
CUDA+NVSHMEM EventTensor component
all-gather and GEMM operator table
```

The all-gather operator decides who owns each K row:

```text
owner_for_k(problem.k, n_pes, kk)
```

If this rank owns the row, it reads local `b`. Otherwise it reads the owning
rank's symmetric `b` through NVSHMEM. The GEMM operator then multiplies `A`
with the completed `gathered_b`.

## 1 Host, 1 GPU Run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target megacu_cuda_allgather_gemm_correctness
ctest --test-dir build \
  -R '^cuda_allgather_gemm_correctness$' \
  --output-on-failure
```

The test runs golden, baseline, host-orch, and seeded-orch and compares all
outputs against the same CPU-side expected GEMM values.

## 1 Host, 2 GPU Run

The two-rank run uses one host with two GPUs:

```text
rank 0 -> GPU 0 -> owns the first K slice of B
rank 1 -> GPU 1 -> owns the second K slice of B
both ranks run all-gather producer tasks
sync tasks wait for dependency groups and EventTensor tile readiness
GEMM consumer tasks compute output tiles from gathered_b
tests compare Megacu output to golden and baseline outputs
```

The checked two-rank smoke intentionally uses odd K: rank 0 owns three K rows
and rank 1 owns two K rows, exercising uneven ownership and `wait_strided`
fan-in.

Manual local setup:

```bash
cmake -S . -B build-nvshmem -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DMEGACU_ENABLE_NVSHMEM_TESTS=ON

cmake --build build-nvshmem \
  --target megacu_nvshmem_two_rank_ag_gemm_smoke \
  --target megacu_mpi_two_rank_ag_gemm_smoke

ctest --test-dir build-nvshmem \
  -R 'nvshmem_two_rank_ag_gemm_correctness_megacu|mpi_two_rank_ag_gemm_correctness_megacu' \
  --output-on-failure
```

## Docker: Full Reproducible Run

```bash
export MEGACU_DOCKER_GPUS='"device=0,1"'
export MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8'
export MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem'

tools/cuda_nvshmem/run_two_card_docker.sh
```

## Docker: Focused AG-GEMM Run

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
      --target megacu_cuda_allgather_gemm_correctness \
      --target megacu_nvshmem_two_rank_ag_gemm_smoke \
      --target megacu_mpi_two_rank_ag_gemm_smoke &&
    ctest --test-dir build-nvshmem \
      -R "cuda_allgather_gemm_correctness|nvshmem_two_rank_ag_gemm_correctness_megacu|mpi_two_rank_ag_gemm_correctness_megacu" \
      --output-on-failure
  '
```

## Torch UID Bootstrap Path

The Docker full gate also launches this example through Torch. Torch does not
change the Megacu recipe. It only starts two processes and broadcasts the
NVSHMEM unique ID needed by the UID-bootstrap smoke binary.

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

To change the dependency fan-in pattern, edit
`common/allgather_gemm_runtime_recipe.cuh`.

To change the all-gather or GEMM math, edit
`megacu/allgather_gemm_megacu.cu`.

Keep `depends_on_many` and `wait_strided` conceptually separate:

```text
depends_on_many -> scheduler readiness
wait_strided    -> EventTensor sync completion
```
