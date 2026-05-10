# GEMM-AllReduce Megacu Tutorial

This tutorial explains the GEMM-AllReduce example as a beginner would run and
modify it. The short version is:

```text
host code calls cuda_nvshmem_gemm_allreduce_host_orch(...)
or cuda_nvshmem_gemm_allreduce_seeded_orch(...)
  -> Megacu builds a small task program from operator invocations
  -> one CUDA mega-kernel runs the runtime loop
  -> the runtime schedules GEMM tiles, waits on EventTensor state, then runs
     AllReduce tiles
```

The example lives in `examples/cuda_nvshmem/gemm_allreduce/`.

## What This Example Computes

The input is a small matrix multiply:

```text
A: M x K
B: K x N
partial = A * B
out = AllReduce(partial across ranks)
```

On one GPU, the "AllReduce" has only one participant, so `out` equals
`A * B`.

On two GPUs, each rank computes its own local `partial`. The AllReduce
consumer reads every rank's `partial` values and writes their sum to `out` on
each rank. The two-rank smoke test initializes rank 0 with `A = 1`, rank 1
with `A = 2`, and `B = 1`, so every output value is checked against `12`.

## Source Map

```text
common/gemm_allreduce.h
  Public target ABI. This is what tests and callers include.

common/gemm_allreduce_runtime_recipe.cuh
  The Megacu program recipe. It submits tasks and EventTensor attrs.

megacu/megacu_gemm_allreduce.cu
  Operator bodies, operator dispatch table, runtime launch, and CUDA
  mega-kernel runtime implementation.

megacu/gemm_allreduce_orchestrate.cc
  Public host-callable host-orch and seeded-orch wrappers.

megacu/gemm_allreduce_arena.cuh
  Host-side arena storage for host-orch.

golden/
  Reference implementation used by correctness tests.

phased/baseline/
  Handwritten manual mega-kernel baseline. This is not the Megacu path.
```

## Megacu Concepts Used Here

### Raw Arguments

The host-callable Megacu functions take CUDA-like arguments directly:

```cpp
cuda_nvshmem_gemm_allreduce_host_orch(
    driver, a, b, partial, out, events, problem);
```

Megacu does not wrap `a`, `b`, `partial`, or `out` in input/output/inout
objects. They are normal pointers. The user is responsible for passing correct
buffers and explicit dependencies.

### Driver

`gemm_ar_driver` is `megacu::cuda_nvshmem::driver_view`. It carries launch and
distributed execution facts:

```text
driver.launch.stream          CUDA stream for launching the one mega-kernel
driver.launch.device_ordinal  CUDA device for this rank
driver.team.team_my_pe        rank inside the NVSHMEM team
driver.team.team_n_pes        number of ranks in the team
driver.team.cuda_device_ordinal
```

The driver is not the algorithm input. The matrices and problem shape are still
passed as normal target arguments.

### Tasks

The recipe submits three logical tasks per output tile:

```cpp
for (int64_t tile = 0; tile < m_tiles * n_tiles; ++tile) {
  auto gemm = orch.submit(
      gemm_tile_produce,
      attrs(single_tile(tile), notify(ready)));
  auto wait_ready = orch.sync(
      attrs(depends_on(gemm), wait(ready, tile)));
  orch.submit(
      allreduce_tile_consume,
      attrs(depends_on(wait_ready), single_tile(tile)));
}
```

The first and third tasks for each tile are operator tasks. The middle task is
sync-only: it has no operator body. It exists so the runtime can wait for that
tile's EventTensor state without coupling event logic into the GEMM or
AllReduce operators. This lets AllReduce for a tile run as soon as that tile's
producer and EventTensor condition are complete.

### EventTensor

The recipe declares one EventTensor called `ready`.

```cpp
auto ready = orch.event_tensor(attrs(
    event_tensor::shape(m_tiles, n_tiles),
    event_tensor::wait_count(driver.team.team_n_pes),
    cuda_nvshmem::event_tensor::symmetric_storage(events),
    cuda_nvshmem::event_tensor::scope::team{}));
```

Each GEMM tile notifies this EventTensor. The sync-only task waits until every
rank has notified the same tile. The operators do not manually signal or wait;
the runtime lowers the attrs through the CUDA+NVSHMEM EventTensor component.

### Scheduler And Dispatcher

The scheduler only looks at explicit dependency attrs such as
`depends_on(gemm)`. It does not inspect pointers or tensor operations.

The dispatcher looks at tile attrs. In this example, `single_tile(tile)` means
each submitted task owns exactly one GEMM output tile.

### Host-Orch And Seeded-Orch

Both execution models run the same recipe and operators.

```text
host-orch:
  host builds task records
  host seals arena
  host launches one CUDA mega-kernel

seeded-orch:
  host seeds raw args and arena storage
  first device control path builds task records inside the mega-kernel
  worker blocks run the same runtime loop
```

## What The CUDA Mega-Kernel Does

`megacu/megacu_gemm_allreduce.cu` links:

```text
runtime::device_persistent
  execution model: host-orch or seeded-orch
  loop: block_tile
  scheduler: explicit_asap
  dispatcher: tile_grid
  EventTensor: CUDA+NVSHMEM int EventTensor
  operators: GEMM tile and AllReduce tile
```

The runtime loop repeatedly:

1. asks the scheduler which tasks are dependency-ready;
2. asks the dispatcher which tile this CUDA block should run;
3. lets EventTensor handle notify/wait attrs;
4. invokes the operator slot when the ready task is an operator task;
5. marks tile work complete.

The CUDA stream only launches and synchronizes the outer mega-kernel. It is not
used as the internal scheduler.

## 1 Host, 1 GPU Run

This path validates local CUDA execution. It runs golden, manual baseline,
Megacu host-orch, and Megacu seeded-orch, then compares numeric output values.

From the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target megacu_cuda_gemm_allreduce_correctness
ctest --test-dir build \
  -R '^cuda_gemm_allreduce_correctness$' \
  --output-on-failure
```

To choose the CUDA device for the local test:

```bash
MEGACU_TEST_CUDA_DEVICE=0 ctest --test-dir build \
  -R '^cuda_gemm_allreduce_correctness$' \
  --output-on-failure
```

## 1 Host, 2 GPU Run

This path starts two NVSHMEM ranks on one host. Each rank chooses a different
CUDA device, allocates symmetric `partial` and EventTensor storage, and calls
the same host-callable Megacu functions.

```text
rank 0 -> GPU 0 -> A values are 1 -> computes partial
rank 1 -> GPU 1 -> A values are 2 -> computes partial
both ranks enter one Megacu mega-kernel
EventTensor waits for both ranks per tile
AllReduce task reads local and remote partial values
both ranks check out == 12 for every element
```

Manual local setup, assuming CUDA and NVSHMEM are already installed:

```bash
cmake -S . -B build-nvshmem -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DMEGACU_ENABLE_NVSHMEM_TESTS=ON

cmake --build build-nvshmem --target megacu_nvshmem_two_rank_smoke

ctest --test-dir build-nvshmem \
  -R 'nvshmem_two_rank_gemm_ar_correctness_(golden|manual_baseline|megacu)' \
  --output-on-failure
```

## Docker: Full Reproducible Run

The Docker image is the expected PR validation environment. It includes CUDA,
NVSHMEM, MPI, PyTorch, and NumPy.

```bash
export MEGACU_DOCKER_GPUS='"device=0,1"'
export MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8'
export MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem'

tools/cuda_nvshmem/run_two_card_docker.sh
```

This builds the image, configures `build-nvshmem`, builds all examples, runs
direct tests, MPI tests, Torch UID-bootstrap tests, and the final CTest filter.

## Docker: Focused GEMM-AllReduce Run

Use this when you only want this example:

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
      --target megacu_cuda_gemm_allreduce_correctness \
      --target megacu_nvshmem_two_rank_smoke &&
    ctest --test-dir build-nvshmem \
      -R "cuda_gemm_allreduce_correctness|nvshmem_two_rank_gemm_ar_correctness_(golden|manual_baseline|megacu)" \
      --output-on-failure
  '
```

## How To Modify The Example

To change the program structure, edit
`common/gemm_allreduce_runtime_recipe.cuh`.

To change what a tile computes, edit `gemm_tile_produce_task` or
`allreduce_tile_consume_task` in `megacu/megacu_gemm_allreduce.cu`.

To add another operator stage:

1. add a new slot in `runtime_slots`;
2. submit it from `build_runtime_recipe`;
3. add an operator functor in `megacu/megacu_gemm_allreduce.cu`;
4. extend `gemm_allreduce_operators::invoke`;
5. update correctness tests to compare against golden output.
