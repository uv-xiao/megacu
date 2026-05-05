# CUDA + NVSHMEM Tiny Decode Pipeline

This example will demonstrate a small decode-style pipeline with multiple
operator stages, explicit readiness, and a Megacu runtime path that is not a
single GEMM-shaped example.

## Layout

- `common/`: shared problem definitions and orchestration declarations.
- `golden/`: correctness reference code.
- `baseline/`: handwritten CUDA comparison code.
- `megacu/`: Megacu-composed operator task path.

## Intended Megacu Flow

```text
orchestrate(driver, args)
  declare EventTensor readiness for stage outputs
  submit norm and projection tile/range tasks
  submit residual or activation tasks after readiness waits
  submit MLP-style tile/range tasks
  submit logits/head tile/range tasks
  launch one runtime-owned CUDA path
```

Scheduler, dispatcher, runtime, EventTensor, platform, and backend components
own the task issue path. Handwritten large fused or persistent kernels are a
baseline-only comparison point. The Megacu path is reserved for composing
tile/range operators as tasks.

## Build And Test

```bash
cmake -S . -B build
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R 'tiny_decode_correctness|^tiny_decode_example_layout$' --output-on-failure
```

The layout CTest checks the example shape. The correctness CTest compares the
golden, handwritten baseline, and Megacu-composed stage path when CUDA language
support is enabled.

## Run

```bash
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R tiny_decode_correctness --output-on-failure
```

The first correctness path is local and host-orchestrated. It does not launch
distributed tiny decode work.

## Assumptions

- CUDA language support is available for the object target.
- The initial tiny decode runner is expected to use a single-host CUDA runtime
  path; distributed tiny decode behavior is not part of this skeleton.
- Numeric validation will require NVIDIA GPUs and a CUDA toolkit. NVSHMEM may
  be needed only if a later distributed tiny decode runner is added.

## Known Limitations

- The current correctness test does not run CUDA kernels or distributed work.
- Full model loading, large decode dimensions, and distributed tiny decode runs
  remain future work.
