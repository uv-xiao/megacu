# CUDA + NVSHMEM AllGather GEMM

This example demonstrates all-gather producer work making input regions
available to GEMM tile consumer work through the CUDA+NVSHMEM backend.

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
  declare EventTensor readiness for gathered peer regions
  for each output N tile:
    submit all-gather tile producer tasks across arbitrary K tiles
    collect returned producer refs in a scheduler dependency group
    submit one sync-only readiness task with depends_on_many + wait_strided
    submit GEMM tile consumer tasks for output M tiles
  launch one runtime-owned CUDA megakernel
```

Scheduler, dispatcher, runtime, EventTensor, platform, and backend components
own the task issue path. Handwritten large fused or persistent kernels are a
baseline-only comparison point. The Megacu path is reserved for composing
tile/range operators as tasks.

## Build And Test

```bash
cmake -S . -B build
cmake --build build --target megacu_cuda_allgather_gemm_correctness
ctest --test-dir build -R 'ag_gemm_example_layout|cuda_allgather_gemm_correctness' --output-on-failure
```

The correctness CTest compares golden, handwritten CUDA baseline, Megacu
host-orch, and Megacu seeded-orch paths for one host and one GPU.

When the distributed NVSHMEM targets are enabled, the two-rank AG-GEMM smokes
cover direct NVSHMEM, MPI, and Torch UID bootstrap launch paths with odd K:
rank 0 owns three K rows and rank 1 owns two K rows.

## Run

```bash
cmake --build build --target megacu_cuda_allgather_gemm_correctness
ctest --test-dir build -R '^cuda_allgather_gemm_correctness$' --output-on-failure
```

## Assumptions

- CUDA language support is available for the numeric targets.
- The distributed backend is CUDA+NVSHMEM with symmetric memory and rank/team
  setup supplied by the eventual runner.
- Numeric validation will require NVIDIA GPUs, a CUDA toolkit, NVSHMEM headers
  and libraries, and an NVSHMEM-capable launcher.

## Known Limitations

- The default correctness path proves one host and one GPU.
- The layout CTest does not run CUDA kernels or NVSHMEM communication.
- Direct, MPI, and Torch launch adapters are exercised through the shared
  CUDA+NVSHMEM helper scripts when the matching distributed runtime is enabled.
