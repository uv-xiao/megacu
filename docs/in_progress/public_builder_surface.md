# Feature Task: Complete First Megacu Implementation

- Branch: `implementation/public-builder-surface`
- PR: #3
- Owner: Codex
- Status: Active

## Goal

Implement the first locally verifiable Megacu lifecycle from the accepted
implementation-ready design: public program authoring, host materialization,
target metadata sections, CMake component/orchestrate target plumbing, direct
compiled orchestrate ABI, runtime validation, and negative overlap-guard checks
for the phased and overlap GEMM+AllReduce targets.

## Input

- `docs/design/implementation_ready_device_native_layer/01-program.md`
- `docs/design/implementation_ready_device_native_layer/07-first-validation-slice.md`
- `docs/design/implementation_ready_device_native_layer/08-verification.md`
- `docs/design/implementation_ready_device_native_layer/10-implementation-architecture.md`

## Output

- Public headers:
  - `include/megacu/program.h`
  - `include/megacu/views.h`
  - `include/megacu/platform/cuda.h`
  - `include/megacu/backends/nvshmem.h`
- Internal first-pass record declarations:
  - `include/megacu/detail/program_ir.h`
- Target metadata and materialization headers:
  - `include/megacu/detail/target_metadata.h`
  - `include/megacu/detail/materialize.h`
- CMake functions:
  - `cmake/MegacuTargets.cmake`
- Example target:
  - `examples/cuda_nvshmem_gemm_allreduce/`
- Tests proving:
  - `gemm_allreduce_phased_program` and `gemm_allreduce_overlap_program` can be
    authored with the public builder surface;
  - materialization emits program facts and metadata sections;
  - overlap schedules require a co-residency guard;
  - direct orchestrate ABI validates typed runtime views and returns
    `megacu::status`;
  - target creation uses CMake/native build plumbing without generated CUDA.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Implement public builder and typed view headers
- [x] Implement minimal internal `program_ir` record declarations
- [x] Add compile-only GEMM+AllReduce descriptor coverage
- [x] Implement materialization and target metadata sections
- [x] Implement CMake component/orchestrate target plumbing
- [x] Implement direct GEMM+AllReduce orchestrate ABI smoke path
- [x] Add negative overlap-guard and metadata validation tests
- [x] Verify locally across build, materialization, metadata, and ABI tests
- [ ] Sync `docs/todo/` and `docs/in_progress/`

## Verification

- Public-header inspection proves no dispatcher, scheduler, kernel-lowering,
  platform, backend, or metadata plan classes are exposed.
- Compile-only test proves phased and overlap GEMM+AllReduce descriptors build
  using only the planned public headers.
- Materializer tests prove extents, domains, participants, resources, events,
  submissions, dispatch, schedule, kernel, and backend sections are present.
- Negative materializer test proves an overlap target with blocking device wait
  and no co-residency guard fails.
- Direct ABI tests prove invalid runtime resources return non-OK
  `megacu::status` before any launch path.
- Static or grep check proves no runtime strategy selection, string event
  lookup, generated CUDA source, or generic `runtime_env` surface is introduced.

## Tests

- Add a focused build/compile-only test target for the two program descriptors.
- Add materialization, metadata validation, CMake, and direct ABI smoke tests.
- Run the repository CMake build and CTest suite.
- If full CUDA/NVSHMEM tooling is unavailable locally, skip only the runtime
  execution checks and record the reason.

## Docs

The accepted design is already in `docs/design/`. This implementation PR should
only update design docs if the code exposes a mismatch in the accepted
contract.

## Closeout

Before closing this task:

- update `docs/todo/README.md` if this completes a listed implementation gap;
- remove this task file or mark follow-up implementation tasks explicitly;
- keep any new public API aligned with the thinness rule in
  `docs/design/implementation_ready_device_native_layer/00-overview.md`.
