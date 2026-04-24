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

## Chosen Targets

The first proof should build two GEMM+AllReduce targets with the same resource
shape and different progress contracts:

- baseline program: `gemm_allreduce_phased_program`;
- overlap program: `gemm_allreduce_overlap_program`;
- platform: CUDA;
- backend: NVSHMEM;
- dispatcher: tiled compute/communication placement;
- baseline scheduler: phased static strategy;
- overlap scheduler: co-resident persistent strategy with a communication
  progress guard;
- kernel lowering: one persistent lowering path that links existing CUDA
  kernels and backend primitives.

The phased target is the first correctness baseline. The overlap target is the
first proof that blocking communication can be made safe by scheduler metadata
that proves producer/consumer co-residency. A payload-copy example would only
prove signaling plumbing.

## Program

The logical program family is `cuda_nvshmem_gemm_allreduce`:

- compute GEMM partial output tiles;
- signal per-tile readiness;
- wait for the same logical tile from all ranks;
- reduce the partial tile through NVSHMEM/multimem or a load-reduce-store
  fallback;
- write the final output tile.

Initial public API proof:

```cpp
struct gemm_allreduce_phased_program;
struct gemm_allreduce_overlap_program;
struct gemm_ar_workspace_slot;
struct event_storage_slot;
struct m_tiles_extent;
struct n_tiles_extent;

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

The matching program descriptors must declare the domains, participants, events,
resources, and submissions shown in `06-examples.md`.

The first proof must show that changing `problem.M`, `problem.N`, and
`problem.K` changes runtime extents inside the precompiled envelope while the
target structure and backend mapping stay fixed by the build graph.

The example must compile without public task descriptors, raw resource ids,
runtime scheduler objects, string-based module loading, string-based event
lookup, or generated per-program CUDA source.

It must also compile without public dispatcher, scheduler, kernel, or backend
plan classes. Those are private metadata sections owned by the selected target
components.

## Runtime Path

The runtime proof should look like:

1. build the reusable component target set;
2. build the concrete phased and overlap GEMM+AllReduce orchestrate targets;
3. construct either an MPI/NVSHMEM session or a Torch/NVSHMEM session;
4. call both compiled orchestrate programs on two ranks/GPUs with explicit
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
  - `include/megacu/detail/target_metadata.h`
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
  - `examples/cuda_nvshmem/gemm_allreduce/`
  - `examples/cuda_nvshmem/gemm_allreduce/torch/`
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
- smoke tests proving invalid runtime resources return non-OK
  `megacu::status` and framework adapters translate that status only at the
  framework boundary;
- linked-artifact and metadata inspection for the built CUDA/NVSHMEM target;
- inspection proving kernel lowering selected/linked existing implementations
  rather than emitting new CUDA/C++ source;
- inspection proving runtime metadata comes from a linked `.megacu.bin` data
  object and contains no raw pointers, spans, string views, type indexes, or
  allocator-owned state;
- linked-symbol inspection proving each CMake `OPS` entry resolved to the
  entrypoint roles required by the selected lowering mode, without requiring
  unused roles;
- smoke tests proving invalid linked metadata returns
  `status_code::metadata_error` before any kernel launch;
- metadata inspection proving the phased target uses `progress_model::phased`;
- metadata inspection proving the overlap target uses
  `progress_model::co_resident_persistent` and contains a residency group with
  both compute and communication workers;
- metadata inspection proving blocking event acquires carry
  `event_wait_mode::blocking_device_wait` and are guarded by either same-group
  residency or an earlier completed phase;
- runtime validation proving symmetric event and partial buffers carry the same
  backend/session identity as the launched NVSHMEM team;
- a negative build or materializer test proving a blocking communication wait
  without a valid co-residency guard fails target lowering;
- MPI-launched two-rank output-correctness test where environment permits;
- Torch-launched two-rank output-correctness test where environment permits;
- explicit skip reason for each unavailable launcher or local NVSHMEM multi-GPU
  execution environment.

Ready-to-implement criteria:

- the public API proof above can be written using only the planned headers;
- the CMake proof can be written using only the two planned CMake functions;
- no public header exposes dispatcher, scheduler, kernel-lowering, platform, or
  backend plan classes;
- any public view is either required by the GEMM+AllReduce target ABI or remains
  target-specific;
- each materialized/lowered metadata section has a single owner from
  `05-dispatcher-scheduler-kernel.md`;
- `run` has no path to build logic or runtime strategy selection;
- the GEMM+AllReduce example has a handwritten CUDA/NVSHMEM baseline or source
  inspection target for comparing the linked execution path.

## Implementation Milestones

The implementation should land in this order:

1. Public builder compile-only surface:
   `program_builder`, tags, domains, participants, resources, events, and
   submissions compile for `gemm_allreduce_phased_program` and
   `gemm_allreduce_overlap_program`.
2. Host materializer:
   `materialize_program<gemm_allreduce_phased_program>()` and
   `materialize_program<gemm_allreduce_overlap_program>()` produce `program_ir`
   and JSON dumps with extents, domains, participants, resources, events, and
   submissions.
3. CMake target plumbing:
   `megacu_add_components(...)` and `megacu_add_orchestrate_target(...)` build
   component targets and run the materializer without generating C++/CUDA.
4. Dispatcher/scheduler/backend metadata:
   the materializer output includes dispatch, schedule, kernel, and backend
   sections for the phased and overlap CUDA/NVSHMEM targets.
5. Linked metadata object:
   CMake embeds each `.megacu.bin` as linked data, and runtime metadata
   validation rejects malformed magic, version, size, checksum, or missing table
   fields.
6. Direct ABI smoke test:
   deployment code can call
   `cuda_nvshmem_gemm_allreduce_phased_orchestrate(...)` and
   `cuda_nvshmem_gemm_allreduce_overlap_orchestrate(...)`
   without constructing an executor or runtime environment.
7. Co-residency guard proof:
   metadata inspection proves the overlap schedule launches compute and
   communication workers in the same residency group, and a negative test proves
   unsafe blocking waits are rejected.
8. MPI/NVSHMEM launch proof:
   `mpirun -np 2 ./cuda_nvshmem_gemm_allreduce_mpi ...` constructs a
   `mpi_nvshmem_session`, validates the two-PE team, and runs correctness where
   hardware exists.
9. Torch/NVSHMEM launch proof:
   `torchrun --standalone --nnodes=1 --nproc-per-node=2 ...` constructs a
   fixed-world optional Torch wrapper, validates Torch rank/world against
   NVSHMEM PE/count, and runs correctness where hardware exists without adding
   Python to Megacu core.

Each milestone should have a small test or inspection artifact before moving to
the next one. New public headers are not allowed between milestones unless the
milestone fails without them and the failure is recorded as evidence. The first
implementation should not start with the MPK-style decode example; that example
is the second validation target once the metadata sections and direct ABI are
working.

## Second Validation Example

After the first proof, the larger example is an MPK-style decode-layer target
from `06-examples.md`. It validates that Megacu can express a real serving
iteration with CUDA-provided tasks such as RMSNorm, linear, paged attention,
split reduction, and residual output without importing MPK's generated-CUDA
pipeline as Megacu's lowering model.
