# First Validation Slice

The first proof should validate the CMake-managed build graph, compiled
orchestrate-program surface, and a real compute/communication fusion shape.

## Goal

Prove all of the following with the smallest useful system:

- named kernels remain native CUDA/NVSHMEM code;
- CMake reusable component targets produce dispatcher/scheduler/kernel-lowering
  engines;
- CMake orchestrate-target compilation reuses those compiled artifacts;
- runtime C++ calls the compiled orchestrate program directly;
- `run(...)` is the repeated fast path inside that program;
- fine-grained GEMM/AllReduce overlap can be expressed without public fragment
  APIs or generated CUDA source.

## Chosen Target

The first proof should build one target:

- program: `gemm_allreduce_program`;
- platform: CUDA;
- backend: NVSHMEM;
- dispatcher: tiled compute/communication placement;
- scheduler: one static persistent strategy;
- kernel lowering: one persistent lowering path that links existing CUDA
  kernels and backend primitives.

This target is narrow enough to implement first, but strong enough to prove the
architecture. A payload-copy example would only prove signaling plumbing.

## Program

The logical program is `cuda_nvshmem_gemm_allreduce`:

- compute GEMM partial output tiles;
- signal per-tile readiness;
- wait for the same logical tile from all ranks;
- reduce the partial tile through NVSHMEM/multimem or a load-reduce-store
  fallback;
- write the final output tile.

Initial public API proof:

```cpp
struct gemm_allreduce_program;
struct gemm_ar_workspace_slot;
struct event_storage_slot;
struct m_tiles_extent;
struct n_tiles_extent;

void cuda_nvshmem_gemm_allreduce_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

The matching program descriptor must declare the domains, participants, events,
resources, and submissions shown in `08-examples.md`.

The first proof must show that changing `problem.M`, `problem.N`, and
`problem.K` changes runtime extents inside the precompiled envelope while the
target structure and backend mapping stay fixed by the build graph.

The example must compile without public task descriptors, raw resource ids,
runtime scheduler objects, string-based module loading, string-based event
lookup, or generated per-program CUDA source.

## Runtime Path

The runtime proof should look like:

1. build the reusable component target set;
2. build the concrete GEMM+AllReduce orchestrate target;
3. construct either an MPI/NVSHMEM session or a Torch/NVSHMEM session;
4. call the compiled orchestrate program on two ranks/GPUs with explicit
   `launch` and `team` views;
5. verify the output against a native baseline for one or more small matrix
   shapes;
6. inspect metadata or linked symbols proving the target uses the selected
   dispatcher, scheduler, lowering, CUDA platform, NVSHMEM backend, and named
   kernel implementations.

The runtime should not call CMake/build logic or select scheduler/backend
strategy.

## Planned Paths

- Public API headers:
  - `include/megacu/program.h`
  - `include/megacu/views.h`
  - `include/megacu/backends/nvshmem.h`
  - `include/megacu/platform/cuda.h`
  - `include/megacu/launch/mpi_nvshmem.h`
- Internal metadata headers:
  - `include/megacu/detail/program_ir.h`
  - `include/megacu/detail/dispatch_plan.h`
  - `include/megacu/detail/schedule_plan.h`
  - `include/megacu/detail/kernel_plan.h`
  - `include/megacu/detail/backend_plan.h`
- Build rules:
  - `cmake/MegacuTargets.cmake`
- Internal implementation:
  - `src/program/`
  - `src/dispatcher/`
  - `src/scheduler/`
  - `src/lowering/`
  - `src/platform/cuda/`
  - `src/backends/nvshmem/`
  - `src/launch/mpi_nvshmem/`
  - `src/integrations/torch/`
  - `src/target/`
- Proof:
  - `examples/cuda_nvshmem_gemm_allreduce/`
  - `tests/build/`
  - `tests/runtime/`
  - `tests/integration/torch/`

## Acceptance Evidence

Required evidence:

- CMake/build log proving the reusable component targets were built once;
- CMake/build log proving the GEMM+AllReduce orchestrate target reused those
  component artifacts;
- compile-only checks for the orchestrate target ABI;
- direct-call smoke test API checks;
- linked-artifact and metadata inspection for the built CUDA/NVSHMEM target;
- inspection proving kernel lowering selected/linked existing implementations
  rather than emitting new CUDA/C++ source;
- MPI-launched two-rank output-correctness test where environment permits;
- Torch-launched two-rank output-correctness test where environment permits;
- explicit skip reason for each unavailable launcher or local NVSHMEM multi-GPU
  execution environment.

Ready-to-implement criteria:

- the public API proof above can be written using only the planned headers;
- the CMake proof can be written using only the two planned CMake functions;
- each materialized/lowered record has a single owner from
  `07-dispatcher-scheduler-kernel.md`;
- `run` has no path to build logic or runtime strategy selection;
- the GEMM+AllReduce example has a handwritten CUDA/NVSHMEM baseline or source
  inspection target for comparing the linked execution path.

## Implementation Milestones

The implementation should land in this order:

1. Public builder compile-only surface:
   `program_builder`, tags, domains, participants, resources, events, and
   submissions compile for `gemm_allreduce_program`.
2. Host materializer:
   `materialize_program<gemm_allreduce_program>()` produces `program_ir` and a
   JSON dump with extents, domains, participants, resources, events, and
   submissions.
3. CMake target plumbing:
   `megacu_add_components(...)` and `megacu_add_orchestrate_target(...)` build
   component targets and run the materializer without generating C++/CUDA.
4. Dispatcher/scheduler/backend metadata:
   the materializer output includes dispatch, schedule, kernel, and backend
   plans for the static CUDA/NVSHMEM target.
5. Direct ABI smoke test:
   deployment code can call
   `cuda_nvshmem_gemm_allreduce_orchestrate(workspace, events, launch, team, problem)`
   without constructing an executor or runtime environment.
6. MPI/NVSHMEM launch proof:
   `mpirun -np 2 ./cuda_nvshmem_gemm_allreduce_mpi ...` constructs a
   `mpi_nvshmem_session`, validates the two-PE team, and runs correctness where
   hardware exists.
7. Torch/NVSHMEM launch proof:
   `torchrun --standalone --nnodes=1 --nproc-per-node=2 ...` constructs a
   fixed-world `NvshmemSession`, validates Torch rank/world against NVSHMEM
   PE/count, and runs correctness where hardware exists.

Each milestone should have a small test or inspection artifact before moving to
the next one. The first implementation should not start with the MPK-style
decode example; that example is the second validation target once the primitive
records and direct ABI are working.

## Second Validation Example

After the first proof, the larger example is an MPK-style decode-layer target
from `08-examples.md`. It validates that Megacu can express a real serving
iteration with CUDA-provided tasks such as RMSNorm, linear, paged attention,
split reduction, and residual output without importing MPK's generated-CUDA
pipeline as Megacu's lowering model.
