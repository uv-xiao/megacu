# Runtime-Linked Device-Native Layer Overview

Status: active redesign after PR #3 review.

This workstream was moved back from `docs/design/` because the previous
implementation-ready design still described a compiler-like path: record a
program IR, materialize owned facts, build dispatch/schedule/kernel/backend
sections, and then validate linked metadata. That is not the Megacu layer we
want.

Megacu is a thin runtime-linked layer:

- CMake links platform/backend/dispatcher/scheduler/operator components.
- The public C++ surface exposes direct orchestrate functions and typed runtime
  views.
- Dispatcher, scheduler, platform, and backend components run at runtime inside
  the linked orchestrate target.
- Kernels/operators are handwritten native implementations linked into the
  target.
- No normal Megacu path builds a program IR, lowers an IR, materializes target
  metadata sections, or generates source.

## Non-Negotiable Correction

The corrected architecture must not include:

- `program_ir` as an implementation dependency;
- `materialize_program`;
- `owned_program_ir`;
- `target_metadata` as the main target contract;
- static `dispatch_section`, `schedule_section`, `kernel_section`, or
  backend-section construction;
- a host materializer executable;
- target lowering as a compiler stage;
- runtime overhead from building intermediate facts before each launch.

The implementation may still expose small typed runtime structs, but only when
they are passed directly to runtime components or kernels.

## Desired Runtime Shape

```text
user/framework/native driver
  calls cuda_nvshmem_gemm_allreduce_*_orchestrate(...)
        |
        v
orchestrate target validates typed views and linked capability config
        |
        v
runtime dispatcher maps problem/team to tile, rank, lane, peer work
        |
        v
runtime scheduler chooses phased or overlap progress actions for this call
        |
        v
platform/backend adapters validate CUDA/NVSHMEM handles and expose primitives
        |
        v
linked native CUDA/NVSHMEM operator implementation runs
```

This is ordinary linked C++/CUDA. The scheduler and dispatcher are components,
not static compiler passes. Runtime work should be compact enough to disappear
into the same loops and launch setup an expert would write by hand.

## Directory Scope

This in-progress design owns the next implementation direction for:

- `include/megacu/`
- `src/dispatcher/`
- `src/scheduler/`
- `src/platform/cuda/`
- `src/backends/nvshmem/`
- `src/target/`
- `cmake/MegacuTargets.cmake`
- `examples/cuda_nvshmem/gemm_allreduce/`
- distributed launch and framework adapters.

## Reading Order

1. `00-overview.md`: the runtime-linking correction.
2. `01-programming-surface.md`: what users write in C++.
3. `02-build-link-config.md`: what CMake links and what it must not do.
4. `03-runtime-components.md`: dispatcher, scheduler, platform, backend, and
   target runtime contracts.
5. `04-distributed-runtime.md`: CUDA+NVSHMEM under MPI, torch-distributed, and
   single-process launch.
6. `05-gemm-allreduce-example.md`: phased and overlap GEMM+AllReduce path.
7. `06-implementation-plan.md`: concrete replacement plan for PR #3.
8. `07-verification.md`: evidence required before claiming a working slice.
