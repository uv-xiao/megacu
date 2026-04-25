# Concrete Implementation Docs

These documents explain the current PR #3 implementation, not the final
accepted architecture. They are intentionally placed under `docs/in_progress/`
because the implementation is still thin and under review.

The scope is concrete:

- files under `include/megacu/`;
- files under `src/`;
- CMake target wiring in `cmake/MegacuTargets.cmake`;
- the CUDA+NVSHMEM GEMM+AllReduce example path at
  `examples/cuda_nvshmem/gemm_allreduce/`;
- the exact call path from an example user call into the current component
  implementation.

The scope is not:

- a replacement for `docs/design/implementation_ready_device_native_layer/`;
- a claim that the current implementation is complete;
- a new architecture proposal detached from current files.

## Reading Order

1. `01-file-map.md`: what each current file owns.
2. `02-build-and-linking.md`: how CMake links components and example targets.
3. `03-runtime-call-path.md`: call-by-call path for a Megacu-based example.
4. `04-component-contracts.md`: contracts and concrete behavior per component.
5. `05-current-thinness-and-gaps.md`: what is still too thin or missing.

## Current Implementation Shape

```text
authoring header
  examples/.../common/gemm_allreduce.h
        |
        v
public Megacu builder API
  include/megacu/program.h
        |
        v
materialized metadata path
  include/megacu/detail/materialize.h
  src/program/materialize.cc
        |
        v
component metadata builders
  src/dispatcher/
  src/scheduler/
  src/lowering/
  src/backends/nvshmem/
        |
        v
linked orchestrate target
  examples/.../phased/megacu/
  examples/.../overlap/megacu/
        |
        v
runtime validation and native call
  src/platform/cuda/validation.cc
  src/backends/nvshmem/validation.cc
  examples/.../common/gemm_allreduce_orchestrate_common.h
  examples/.../phased|overlap/megacu/*.cu
```

The current implementation already has named component files and inspectable
metadata sections. It does not yet fully execute from the metadata schedule
payload at runtime; the example runtime still routes through a hand-written
native CUDA/NVSHMEM implementation selected by the orchestrate wrapper.
