# CUDA + NVSHMEM GEMM Reduce-Scatter

This example will demonstrate GEMM producer tiles feeding reduce-scatter
consumer work through the CUDA+NVSHMEM backend.

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
  submit sync-only readiness tasks that wait on EventTensor state
  submit reduce-scatter tile or range consumer tasks
  launch one runtime-owned CUDA+NVSHMEM path
```

Scheduler, dispatcher, runtime, EventTensor, platform, and backend components
own the task issue path. Handwritten large fused or persistent kernels are a
baseline-only comparison point. The Megacu path is reserved for composing
tile/range operators as tasks.

## Build And Test

```bash
cmake -S . -B build
cmake --build build --target cuda_nvshmem_gemm_reduce_scatter_megacu
ctest --test-dir build -R '^gemm_rs_example_layout$' --output-on-failure
```

The CTest is a layout check. The build target verifies that the skeleton Megacu
object target is present when CUDA language support is enabled.

## Run

This skeleton has no correctness binary yet. A future runnable path should live
under `examples/cuda_nvshmem/gemm_reduce_scatter/megacu/` and provide a
GEMM-RS correctness entrypoint for local and distributed launch modes.

```bash
cmake --build build --target cuda_nvshmem_gemm_reduce_scatter_megacu
```

There is no executable run command for this skeleton-only step.

## Assumptions

- CUDA language support is available for the object target.
- The distributed backend is CUDA+NVSHMEM with symmetric memory and rank/team
  setup supplied by the eventual runner.
- Numeric validation will require NVIDIA GPUs, a CUDA toolkit, NVSHMEM headers
  and libraries, and an NVSHMEM-capable launcher.

## Known Limitations

- The current skeleton does not implement numeric correctness.
- The layout CTest does not run CUDA kernels or NVSHMEM communication.
- Golden and baseline behavior is planned but not proved by this README.
- Direct, MPI, and Torch launch adapters are required by the current PR and are
  exercised through the shared CUDA+NVSHMEM helper scripts; this skeleton does
  not yet add a numeric GEMM-RS distributed run.
