# CUDA + NVSHMEM Tiny Decode Pipeline

This example demonstrates a small decode-style pipeline with multiple operator
stages, explicit readiness, and a Megacu runtime path that is not a single
GEMM-shaped example.

For a beginner-friendly walkthrough with 1-host-1-GPU, 1-host-2-GPU, and Docker
commands, see [TUTORIAL.md](TUTORIAL.md).

## Layout

- `common/`: shared problem definitions and orchestration declarations.
- `golden/`: correctness reference code.
- `baseline/`: handwritten CUDA comparison code.
- `megacu/`: Megacu-composed operator task path.

## Intended Megacu Flow

```text
orchestrate(driver, args)
  declare EventTensor readiness for stage outputs
  submit norm range tasks
  submit sync-only EventTensor wait task
  submit projection range tasks
  submit sync-only EventTensor wait task
  submit residual range tasks
  submit sync-only EventTensor wait task
  submit MLP range tasks
  submit sync-only EventTensor wait task
  submit logits/head range tasks
  launch one runtime-owned CUDA megakernel
```

Scheduler, dispatcher, runtime, EventTensor, platform, and backend components
own the task issue path. Handwritten large fused or persistent kernels are a
baseline-only comparison point. The Megacu path is reserved for composing
tile/range operators as tasks.

The checked-in Megacu path exposes both runtime execution models:

- `megacu_tiny_decode_megacu_host_orch`: the host builds and seals the arena,
  then the common runtime loop executes it in one CUDA megakernel.
- `megacu_tiny_decode_megacu_seeded_orch`: the host seeds the launch, then a
  device orch path publishes and seals the same recipe inside the megakernel.

## Build And Test

```bash
cmake -S . -B build
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R 'tiny_decode_correctness|^tiny_decode_example_layout$' --output-on-failure
```

The layout CTest checks the example shape. The correctness CTest compares the
golden, handwritten baseline, Megacu host-orch, and Megacu seeded-orch stage
paths when CUDA language support is enabled.

## Run

```bash
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R tiny_decode_correctness --output-on-failure
```

The first correctness path is local and launches CUDA megakernels for both
Megacu runtime execution models. The two-rank MPI smoke launches the same
runtime paths on two local GPUs and compares each rank against golden output.

## Assumptions

- CUDA language support is available for the object target.
- The tiny decode two-rank path is a distributed launch smoke, not a remote
  communication algorithm.
- Numeric validation will require NVIDIA GPUs and a CUDA toolkit. NVSHMEM may
  be needed only by the shared Docker image and other CUDA+NVSHMEM examples.

## Known Limitations

- The two-rank tiny decode path validates distributed launch and per-rank
  correctness, but it does not perform remote reads or writes.
- Full model loading, large decode dimensions, and distributed tiny decode runs
  remain future work.
