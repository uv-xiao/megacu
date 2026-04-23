# First Validation Slice

The first proof should validate the CMake-managed build graph and compiled
orchestrate-program surface.

## Goal

Prove all of the following with the smallest useful system:

- named kernels remain native CUDA/NVSHMEM code
- CMake reusable component targets produce dispatcher/scheduler/kernel-lowering
  engines
- CMake orchestrate-target compilation reuses those compiled artifacts
- runtime C++ calls the compiled orchestrate program directly
- `run(...)` is the repeated fast path inside that program

## Chosen Target

The first proof should build one narrow target:

- platform: CUDA
- backend: NVSHMEM
- dispatcher: simple tile/rank mapping
- scheduler: one static persistent strategy
- kernel lowering: one persistent lowering path

This target is enough to prove the architecture without reopening runtime
strategy choice.

## Program

The logical program should still be the `cuda_nvshmem_event_copy` shape:

- write payload
- signal remote event
- wait remote event
- consume and verify payload

But the authored object is now the orchestrate program itself.

## Runtime Path

The runtime proof should look like:

1. build the reusable component target set
2. build the concrete orchestrate target
3. call the compiled orchestrate program on two ranks/GPUs
4. verify payload visibility

The runtime should not call CMake/build logic or select scheduler/backend
strategy.

## Acceptance Evidence

Required evidence:

- CMake/build log proving the reusable component targets were built once
- CMake/build log proving the orchestrate target reused those component
  artifacts
- compile-only checks for the orchestrate target ABI
- direct-call smoke test API checks
- generated-code inspection for the built CUDA/NVSHMEM target
- two-rank payload visibility test where environment permits
- explicit skip reason where local NVSHMEM multi-GPU execution is unavailable
