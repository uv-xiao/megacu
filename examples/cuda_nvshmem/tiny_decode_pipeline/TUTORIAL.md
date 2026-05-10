# Tiny Decode Pipeline Megacu Tutorial

This tutorial explains the tiny decode pipeline example. It is intentionally
not a GEMM communication example. Its purpose is to show that the same Megacu
runtime model can compose a normal multi-stage operator pipeline.

The example lives in `examples/cuda_nvshmem/tiny_decode_pipeline/`.

## What This Example Computes

The data structure is tiny enough to read directly:

```cpp
struct buffers {
  float hidden[4];
  float norm[4];
  float projection[4];
  float residual[4];
  float mlp[4];
  float logits[3];
};
```

The pipeline is:

```text
hidden
  -> norm
  -> projection
  -> residual
  -> mlp
  -> logits
```

The exact math is deliberately simple:

```text
norm[i]       = hidden[i] * 0.5
projection[i] = norm[i] + (i + 1)
residual[i]   = projection[i] + hidden[i]
mlp[i]        = residual[i]^2 * 0.25
logits[v]     = weighted sum of mlp
```

Golden, baseline, host-orch, and seeded-orch all compute the same buffers, and
the correctness test compares the full buffers with `nearly_equal`.

## Source Map

```text
common/tiny_decode.h
  Public buffer type and entrypoint declarations.

golden/tiny_decode_golden.cc
  CPU/reference path.

baseline/tiny_decode_baseline.cu
  Handwritten CUDA comparison path.

megacu/tiny_decode_runtime_recipe.cuh
  Megacu task and EventTensor recipe.

megacu/tiny_decode_megacu.cu
  CUDA operator bodies and runtime launch.

megacu/tiny_decode_arena.cuh
  Arena capacities and host arena declarations.
```

## Megacu Features Used

### Stage Tasks

Each stage is an operator slot:

```text
norm       slot 1
projection slot 2
residual   slot 3
mlp        slot 4
logits     slot 5
```

Each operator receives a tile id. For example, the norm task writes one hidden
element per tile id.

### EventTensor Between Stages

Each stage except logits notifies a local CUDA EventTensor. The next stage is
guarded by a sync-only task.

```text
norm task
  -> notify norm_ready
sync task
  -> depends_on(norm)
  -> wait(norm_ready)
projection task
  -> depends_on(norm_sync)
```

The operator code does not manually wait on events. It only computes its
assigned element. Event logic is owned by the Megacu runtime component.

### Local CUDA EventTensor

This example uses `megacu::platform::cuda::local_event_tensor_i32`, not the
NVSHMEM EventTensor. That is why it is a good beginner example: it shows
EventTensor behavior without distributed communication.

## What The CUDA Mega-Kernel Does

`tiny_decode_megacu.cu` links:

```text
runtime::device_persistent
  host-orch or seeded-orch execution model
  block_tile loop
  explicit_asap scheduler
  tile_grid dispatcher
  local CUDA EventTensor component
  norm/projection/residual/mlp/logits operator table
```

The runtime loop runs inside one CUDA mega-kernel. It repeatedly asks which
task is ready, maps CUDA blocks to tile work, runs the operator, and completes
sync-only EventTensor tasks.

## 1 Host, 1 GPU Run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build \
  -R '^tiny_decode_correctness$' \
  --output-on-failure
```

This single-process test runs:

```text
golden
baseline
Megacu host-orch
Megacu seeded-orch
Megacu compatibility entry
```

and compares every buffer.

## 1 Host, 2 GPU Run

The tiny decode two-rank path is a distributed launch smoke, not a distributed
communication algorithm. MPI starts two ranks on one host, rank 0 selects GPU
0, rank 1 selects GPU 1, and each rank independently runs the same local tiny
decode Megacu path against golden output.

```text
rank 0 -> GPU 0 -> local tiny decode -> compare with golden
rank 1 -> GPU 1 -> local tiny decode -> compare with golden
```

This proves the runtime execution models can be launched in a two-rank
environment. It does not use NVSHMEM remote reads or writes.

Manual local setup:

```bash
cmake -S . -B build-nvshmem -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DMEGACU_ENABLE_NVSHMEM_TESTS=ON

cmake --build build-nvshmem --target mpi_two_rank_tiny_decode_smoke

ctest --test-dir build-nvshmem \
  -R 'mpi_two_rank_tiny_decode_correctness_megacu' \
  --output-on-failure
```

## Docker: Full Reproducible Run

```bash
export MEGACU_DOCKER_GPUS='"device=0,1"'
export MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8'
export MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem'

tools/cuda_nvshmem/run_two_card_docker.sh
```

## Docker: Focused Tiny Decode Run

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
      --target cuda_nvshmem_tiny_decode_correctness \
      --target mpi_two_rank_tiny_decode_smoke &&
    ctest --test-dir build-nvshmem \
      -R "tiny_decode_correctness|mpi_two_rank_tiny_decode_correctness_megacu" \
      --output-on-failure
  '
```

## How To Modify The Example

To add a new stage:

1. add a new slot in `tiny_decode_runtime::slots`;
2. declare any new EventTensor in `build_recipe`;
3. submit the new operator task with explicit `depends_on` attrs;
4. add the CUDA operator functor in `tiny_decode_megacu.cu`;
5. extend `tiny_decode_operators::invoke`;
6. update golden and baseline output checks.

The key design rule is that stage ordering is explicit in the recipe. Megacu
will not infer that `projection` depends on `norm` by looking at buffer names.
