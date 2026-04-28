# Tiny GEMM+AllReduce Proof Example

The PR #4 example is intentionally tiny. Its job is to prove the
runtime-linked architecture path, not to become the general CUDA+NVSHMEM
implementation.

The example may be problem-specific. It may use one fixed dtype, one small
shape family, one operator symbol, and example-local mapping/scheduling helpers
if that makes the architecture easier to inspect. Those shortcuts must stay out
of public Megacu APIs and shared component contracts.

## What Makes It Mighty

Even though the example is tiny, it should cross the important architectural
boundaries:

```text
direct orchestrate ABI
  -> typed runtime views
  -> linked capability for this target
  -> runtime validation
  -> problem/team values mapped to work
  -> progress policy calls linked native operator
  -> status returns to caller
```

The example proves that Megacu is ordinary linked C++/CUDA runtime code. It
must not prove that the final dispatcher, scheduler, CMake helpers, distributed
adapters, or GEMM+AllReduce suite are general.

## Minimal Target Shape

One direct ABI is enough:

```cpp
megacu::status cuda_nvshmem_tiny_gemm_allreduce_orchestrate(
    tiny_gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    tiny_gemm_ar_problem problem);
```

The problem can be narrow:

- f32 row-major input and output;
- one local GEMM tile or a tiny fixed tile grid;
- `team_n_pes == 1` as the minimum proof, with `team_n_pes == 2` optional if
  it helps validate the distributed boundary;
- a single linked native operator symbol.

The tiny proof should still name where a future general implementation will
replace shortcuts:

- example-local mapping becomes the general annotation-driven dispatcher;
- example-local progress helper becomes scheduler runtime;
- direct CMake wiring becomes reusable `ConfigureTarget` and `OrchTarget`
  helpers;
- narrow validation becomes platform/backend validation contracts.

## Golden, Baseline, And Megacu Roles

For PR #4, keep roles simple:

- `golden`: local expected result generation for the tiny shape;
- `baseline`: optional handwritten CUDA/NVSHMEM comparison only if it clarifies
  the proof;
- `megacu`: the direct runtime-linked Megacu path.

The full future suite remains TODO work:

```text
examples/cuda_nvshmem/gemm_allreduce/
  common/
  golden/
  phased/baseline/
  phased/megacu/
  overlap/baseline/
  overlap/megacu/
```

That full layout belongs with `docs/todo/concrete_impl/` because it requires
stronger implementation generality and broader verification.

## Concrete Tiny Call Path

```text
user/test code
  -> cuda_nvshmem_tiny_gemm_allreduce_orchestrate(...)
  -> make runtime_context from launch/team/events/capability
  -> validate target capability and runtime views
  -> map this tiny problem/team to local work
  -> run the tiny progress policy
  -> call the linked native operator symbol
  -> return megacu::status
```

The mapping step may be problem-specific in PR #4. The required architectural
property is that it happens at runtime from concrete problem/team values, not
from generated static dispatch or schedule sections.

## Rejected Path

The tiny example must not take this path:

```text
program_builder
  -> program_ir
  -> materialize_program
  -> dispatch_section/schedule_section/kernel_section/backend_section
  -> generated or materialized target metadata
  -> example glue selects native symbol
```

Those steps are the architecture being rejected.

## Evidence Expected From The Tiny Example

The tiny example should provide evidence for:

- the direct ABI can validate bad runtime views and return `megacu::status`;
- the linked native operator path is called without a materializer;
- problem/team values are consumed at runtime;
- any problem-specific shortcut is documented as example-local;
- no generated metadata section is required for the example to run.

Numeric CUDA/NVSHMEM correctness, two-card baseline parity, overlap progress,
and framework launch adapters are valuable future evidence, but they are not
the PR #4 gate unless explicitly added back to scope.
