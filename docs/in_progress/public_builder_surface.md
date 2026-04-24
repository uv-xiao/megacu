# Feature Task: Complete First Megacu Implementation

- Branch: `implementation/public-builder-surface`
- PR: #3
- Owner: Codex
- Status: Active

## Goal

Implement the first locally verifiable Megacu lifecycle from the accepted
implementation-ready design: public program authoring, host materialization,
target metadata sections, CMake component/orchestrate target plumbing, direct
compiled orchestrate ABI, runtime validation, numeric GEMM+AllReduce
correctness, and negative overlap-guard checks for the phased and overlap
GEMM+AllReduce targets.

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
  - `examples/cuda_nvshmem/gemm_allreduce/`
  - `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`
- Tests proving:
  - `gemm_allreduce_phased_program` and `gemm_allreduce_overlap_program` can be
    authored with the public builder surface;
  - materialization emits program facts and metadata sections;
  - overlap schedules require a co-residency guard;
  - direct orchestrate ABI validates typed runtime views and returns
    `megacu::status`;
  - CUDA runtime tests exercise a real device stream, device allocation,
    async device operation, and direct orchestrate calls;
  - CUDA multi-card tests exercise two visible GPUs on one host, peer-copy
    runtime plumbing, and one logical PE/orchestrate call per device;
  - Docker NVSHMEM tests exercise two single-host PEs, NVSHMEM host bootstrap,
    symmetric allocation, CUDA stream/device setup, and the direct orchestrate
    ABI with NVSHMEM-backed symmetric views;
  - numeric CUDA tests execute golden phased, golden overlap, Megacu phased,
    and Megacu overlap on a real CUDA stream and compare output matrices
    against host-computed expected values;
  - numeric Docker NVSHMEM tests execute golden phased, golden overlap, Megacu
    phased, and Megacu overlap under two ranks and compare both rank outputs
    after an NVSHMEM sum-reduce;
  - resource arguments bind concrete workspace fields through member pointers,
    so the program IR records both the workspace slot and field name used by
    each kernel argument;
  - phased and overlap static libraries each export a linked metadata symbol
    validated by a consuming test binary;
  - target creation uses CMake/native build plumbing without generated CUDA.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Implement public builder and typed view headers
- [x] Implement minimal internal `program_ir` record declarations
- [x] Add compile-only GEMM+AllReduce descriptor coverage
- [x] Implement materialization and target metadata sections
- [x] Implement CMake component/orchestrate target plumbing
- [x] Implement direct GEMM+AllReduce orchestrate ABI smoke path
- [x] Add real CUDA single-device runtime smoke coverage
- [x] Add single-host two-card CUDA runtime smoke coverage
- [x] Add negative overlap-guard and metadata validation tests
- [x] Verify locally across build, materialization, metadata, ABI, CUDA device,
  and CUDA two-card tests
- [x] Add Docker-provisioned two-card NVSHMEM runtime validation
- [x] Add numeric GEMM+AllReduce correctness on CUDA and two-rank NVSHMEM
- [x] Replace the minimal numeric path with golden phased/overlap baselines and
  two Megacu implementations
- [x] Move example target ownership into the example-local `CMakeLists.txt`
- [x] Make resource-field bindings concrete in program IR
- [x] Prove linked per-target metadata symbols from consuming binaries
- [x] Sync `docs/todo/` and `docs/in_progress/`

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
- CUDA runtime smoke tests prove the local build can touch a real CUDA device
  through `CUDA::cudart`, create a stream, allocate device buffers, perform an
  async device operation, and pass CUDA-backed views through the compiled
  orchestrate ABI.
- CUDA multi-card smoke tests prove the local build can touch two GPUs on the
  same host, perform a peer copy between them, and pass one logical two-PE team
  view per device through the compiled orchestrate ABI.
- CUDA numeric correctness tests prove golden phased, golden overlap, Megacu
  phased, and Megacu overlap run on a real CUDA stream and write expected output
  matrices. The overlap golden uses a persistent fused CUDA kernel with
  co-resident compute and communication CTAs.
- Linked metadata tests prove a consuming binary can link the phased and
  overlap target metadata symbols and validate their schedule mode, extent,
  domain, and team-size facts without re-materializing descriptors.
- Static or grep check proves no runtime strategy selection, string event
  lookup, generated CUDA source, or generic `runtime_env` surface is introduced.
- Host inspection currently finds CUDA, Open MPI, Docker, and two idle A100s,
  but no host NVSHMEM headers, libraries, or launcher. Two-rank NVSHMEM
  validation is therefore provided by a CUDA/NVSHMEM Docker environment.
- Docker NVSHMEM smoke tests prove the packaged NVSHMEM host runtime can
  bootstrap two PEs on one host, allocate symmetric buffers, synchronize both
  PEs, pass NVSHMEM-backed symmetric views into the compiled orchestrate ABI on
  two visible GPUs, execute golden phased/overlap and Megacu phased/overlap
  paths, and verify both ranks observe the expected sum-reduced matrix.

## Tests

- Add a focused build/compile-only test target for the two program descriptors.
- Add materialization, metadata validation, CMake, and direct ABI smoke tests.
- Add CUDA runtime smoke tests for one device and single-host two-card plumbing.
- Add numeric CUDA GEMM+AllReduce correctness tests for the compiled target.
- Run the repository CMake build and CTest suite.
- If CUDA runtime is unavailable, skip only CUDA tests by absence of
  `CUDAToolkit`; if fewer than two GPUs are visible, skip only the two-card
  smoke with CTest skip code `77`.
- If host NVSHMEM tooling is unavailable, use a Docker-provisioned NVSHMEM
  environment for the two-rank single-host validation and record any remaining
  skip reason explicitly.

Current local evidence:

```sh
MEGACU_TEST_CUDA_DEVICES=5,6 MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build --output-on-failure
tools/cuda_nvshmem/gemm_allreduce/run_two_card_docker.sh
```

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
