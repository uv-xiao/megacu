# Feature Task: Public Builder Surface

- Branch: `implementation/public-builder-surface`
- PR: #3
- Owner: Codex
- Status: Active

## Goal

Implement the first compile-only public Megacu programming surface from the
accepted implementation-ready design: `program_builder`, typed tags, domains,
participants, resources, events, submissions, runtime views, and status types
needed by the GEMM+AllReduce phased and overlap program descriptors.

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
- Compile-only examples or tests proving
  `gemm_allreduce_phased_program` and `gemm_allreduce_overlap_program` can be
  authored with the public builder surface.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [ ] Implement public builder and typed view headers
- [ ] Implement minimal internal `program_ir` record declarations
- [ ] Add compile-only GEMM+AllReduce descriptor coverage
- [ ] Verify locally
- [ ] Sync `docs/todo/` and `docs/in_progress/`

## Verification

- Public-header inspection proves no dispatcher, scheduler, kernel-lowering,
  platform, backend, or metadata plan classes are exposed.
- Compile-only test proves phased and overlap GEMM+AllReduce descriptors build
  using only the planned public headers.
- Static or grep check proves no runtime strategy selection, string event
  lookup, generated CUDA source, or generic `runtime_env` surface is introduced.

## Tests

- Add a focused build/compile-only test target for the two program descriptors.
- Run the repository build or the narrow compile-only target once build tooling
  exists.
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
