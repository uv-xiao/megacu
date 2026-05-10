# CUDA + NVSHMEM GEMM Reduce-Scatter

This example demonstrates GEMM producer tiles feeding reduce-scatter consumer
work through the CUDA+NVSHMEM backend.

For a beginner-friendly walkthrough with 1-host-1-GPU, 1-host-2-GPU, MPI,
Torch, and Docker commands, see [TUTORIAL.md](TUTORIAL.md).

## Layout

- `common/`: shared problem definitions and orchestration declarations.
- `golden/`: correctness reference code.
- `baseline/`: handwritten CUDA+NVSHMEM comparison code.
- `megacu/`: Megacu-composed operator task path.

## Intended Megacu Flow

```text
orchestrate(driver, args)
  declare EventTensor readiness for produced GEMM tiles
  submit GEMM tile producer tasks
  submit sync-only readiness task that waits on EventTensor state
  submit reduce-scatter tile or range consumer tasks
  launch one runtime-owned CUDA megakernel
```

Scheduler, dispatcher, runtime, EventTensor, platform, and backend components
own the task issue path. Handwritten large fused or persistent kernels are a
baseline-only comparison point. The Megacu path is reserved for composing
tile/range operators as tasks.

## Build And Test

```bash
cmake -S . -B build
cmake --build build --target megacu_cuda_gemm_reduce_scatter_correctness
ctest --test-dir build -R 'gemm_rs_example_layout|cuda_gemm_reduce_scatter_correctness' --output-on-failure
```

The correctness CTest compares golden, handwritten CUDA baseline, Megacu
host-orch, and Megacu seeded-orch paths for one host and one GPU.

## Run

```bash
cmake --build build --target megacu_cuda_gemm_reduce_scatter_correctness
ctest --test-dir build -R '^cuda_gemm_reduce_scatter_correctness$' --output-on-failure
```

## Assumptions

- CUDA language support is available for the numeric targets.
- The distributed backend is CUDA+NVSHMEM with symmetric memory and rank/team
  setup supplied by the eventual runner.
- Numeric validation will require NVIDIA GPUs, a CUDA toolkit, NVSHMEM headers
  and libraries, and an NVSHMEM-capable launcher.

## Known Limitations

- The layout CTest only checks repository shape.
- Numeric correctness is covered by the CUDA one-GPU test and the two-rank
  CUDA+NVSHMEM/MPI/Torch smoke paths when the matching runtime dependencies are
  enabled.
