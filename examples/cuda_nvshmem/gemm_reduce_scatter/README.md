# CUDA + NVSHMEM GEMM Reduce-Scatter

This example demonstrates GEMM producer tiles feeding reduce-scatter consumer
work through the CUDA+NVSHMEM backend.

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

- The current correctness path proves one host and one GPU only.
- The layout CTest does not run CUDA kernels or NVSHMEM communication.
- Direct, MPI, and Torch launch adapters are required by the current PR and are
  exercised through the shared CUDA+NVSHMEM helper scripts; the distributed
  numeric GEMM-RS run remains pending.
