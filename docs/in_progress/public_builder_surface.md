# Feature Task: First Working Megacu Slice

- Branch: `implementation/public-builder-surface`
- PR: #3
- Owner: Codex
- Status: Active, scope reset after PR review

## Review Reset

The PR cannot be treated as a working Megacu implementation while dispatcher,
scheduler, kernel-lowering, platform/backend binding, and target-lowering are
only metadata labels or example-local shortcuts. A correct first slice must run
through the same component boundaries described by the accepted design and then
prove the resulting target on CUDA and CUDA+NVSHMEM.

The actionable review input for this reset came from a pending GitHub review.
Normal submitted-comment endpoints returned no comments, but
`gh api repos/uv-xiao/megacu/pulls/3/reviews/4175138919/comments` exposed the
pending comments. The review says the PR lacks real dispatch/schedule/etc.,
does not look like a working version, has questionable file organization, mixes
phased and overlap examples, treats the overlap golden as fake, keeps numeric
validation in orchestrate code, and puts orchestrate code inside tests.

## Design Contract

This task implements the first complete local lifecycle from
`docs/design/implementation_ready_device_native_layer/`:

- public C++ authoring records the logical program;
- materialization converts the program into owned target facts;
- dispatcher maps logical domains and virtual participants to execution
  placements;
- scheduler turns dispatch output plus events into phased or co-resident
  persistent execution;
- kernel lowering links existing CUDA/NVSHMEM implementations to named ops and
  selected schedule payloads without emitting generated CUDA/C++ source;
- CUDA platform and NVSHMEM backend adapters validate runtime views, bind native
  launch/team/event resources, and expose device-side backend primitives;
- CMake builds reusable components and then links concrete orchestrate targets;
- runtime code calls the compiled orchestrate functions directly.

The public surface must stay thin: public headers may expose only authoring
helpers, typed runtime views, CMake functions, and the direct orchestrate ABI.
Dispatcher, scheduler, lowering, platform, backend, and metadata sections are
private implementation details.

## Required File Organization

Core implementation must not live primarily inside the example. The expected
ownership is:

- `include/megacu/`: public C++ authoring and runtime view API only.
- `include/megacu/detail/`: private record and metadata declarations used by
  tests and target plumbing.
- `src/program/`: program materialization and validation.
- `src/dispatcher/`: tiled compute/communication dispatcher implementation.
- `src/scheduler/`: phased and co-resident persistent scheduler
  implementation, including overlap guard validation.
- `src/lowering/`: kernel and target lowering driver that links existing
  implementations and builds launch payloads.
- `src/platform/cuda/`: CUDA stream/device validation and launch adapter.
- `src/backends/nvshmem/`: NVSHMEM team, symmetric resource, peer, and
  device-side event/communication adapter.
- `src/target/`: linked metadata object validation and target runtime ABI
  helpers.
- `examples/cuda_nvshmem/gemm_allreduce/common/`: shared descriptors and
  runtime helpers for the GEMM+AllReduce example family.
- `examples/cuda_nvshmem/gemm_allreduce/golden/`: Megacu-free local golden
  result generation and pure CUDA/NVSHMEM baseline entrypoints.
- `examples/cuda_nvshmem/gemm_allreduce/phased/baseline/`: phased baseline
  ownership point.
- `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/`: phased Megacu
  descriptors, kernels, README, and example-owned CMake.
- `examples/cuda_nvshmem/gemm_allreduce/overlap/baseline/`: overlap baseline
  ownership point.
- `examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/`: overlap Megacu
  descriptors, kernels, README, and example-owned CMake.
- `docker/cuda_nvshmem/`: shared CUDA+NVSHMEM Docker support. Add per-example
  Docker assets only if an example genuinely needs a unique image.
- `tools/cuda_nvshmem/`: shared CUDA+NVSHMEM run support. Add per-example tools
  only if an example genuinely needs unique scripts.

Example code may provide target-specific kernels and descriptors. It must not
be the only place where dispatch, scheduling, lowering, CUDA platform, or
NVSHMEM backend behavior exists.

## Working Slice Outputs

The PR should produce these implementation artifacts:

- Public headers:
  - `include/megacu/program.h`
  - `include/megacu/views.h`
  - `include/megacu/platform/cuda.h`
  - `include/megacu/backends/nvshmem.h`
- Private records:
  - `include/megacu/detail/program_ir.h`
  - `include/megacu/detail/target_metadata.h`
  - `include/megacu/detail/materialize.h`
- Reusable component implementations under `src/program/`, `src/dispatcher/`,
  `src/scheduler/`, `src/lowering/`, `src/platform/cuda/`,
  `src/backends/nvshmem/`, and `src/target/`.
- CMake functions in `cmake/MegacuTargets.cmake` that compile reusable
  components and link concrete orchestrate targets without generating CUDA.
- One GEMM+AllReduce example family with split variant implementation
  directories:
  - `examples/cuda_nvshmem/gemm_allreduce/common/`
  - `examples/cuda_nvshmem/gemm_allreduce/golden/`
  - `examples/cuda_nvshmem/gemm_allreduce/phased/baseline/`
  - `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/`
  - `examples/cuda_nvshmem/gemm_allreduce/overlap/baseline/`
  - `examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/`
- Two Megacu orchestrate targets for one design family:
  - `cuda_nvshmem_gemm_allreduce_phased_orchestrate`
  - `cuda_nvshmem_gemm_allreduce_overlap_orchestrate`
- Local golden result generation:
  - `golden_local_gemm`, which is just local GEMM used to produce expected
    numeric results.
- Four Megacu-free CUDA+NVSHMEM baselines:
  - `baseline_phased_single_card`
  - `baseline_phased_multi_card`
  - `baseline_overlap_single_card`
  - `baseline_overlap_multi_card`
- Two Megacu implementations:
  - `megacu_phased`
  - `megacu_overlap`
- Single-card and two-card execution for both Megacu implementations is
  selected from runtime team/backend capability rather than duplicated program
  designs.

## Component Requirements

### Program And Materialization

- Authoring must collect typed extents, domains, participants, resources,
  events, named ops, submissions, and resource-field bindings.
- Materialization must own copied records, reject missing required tables, and
  feed dispatcher/scheduler/lowering inputs.
- String names may be diagnostic labels only. Runtime event/op/resource lookup
  must use typed slots or materialized indexes.

### Dispatcher

- Implement a real tiled compute/communication dispatcher for GEMM+AllReduce.
- Output must identify compute work, communication work, tile coordinates,
  participant slots, logical ranks, worker roles, and backend peers.
- Virtual participant mapping belongs to the dispatcher policy, not CMake
  ad hoc mapping syntax.
- Single-card and two-card cases should share the same logical program and
  diverge only through dispatcher/backend capability inputs.

### Scheduler

- Implement phased static scheduling for the baseline target.
- Implement co-resident persistent scheduling for the overlap target.
- The overlap scheduler must materialize residency groups and reject blocking
  communication waits unless compute and communication workers are proven to be
  launched together in the same residency group.
- The phased scheduler may allow GEMM tile production and reduction tile work
  to proceed as soon as dependencies are satisfied; it should not introduce
  false whole-program dependencies unless the backend capability requires them.

### Kernel And Target Lowering

- Lowering links existing CUDA/NVSHMEM symbols and reusable component code; it
  does not generate per-program CUDA/C++ source.
- Lowering must bind named ops to required entrypoint roles for the selected
  strategy, validate missing symbols, and record inspectable kernel metadata.
- The overlap path must use a persistent loop for co-resident compute and
  communication workers.
- The phased path may use separate launches or a persistent phase loop, but it
  must still run through Megacu dispatch and schedule payloads.

### CUDA Platform And NVSHMEM Backend

- CUDA adapter validates device ordinal, stream, launch shape, cooperative or
  persistent launch requirements, and device capability constraints.
- NVSHMEM adapter validates team size, local PE, symmetric event storage,
  symmetric partial buffers, session identity, and peer mapping.
- Multi-card communication in both baseline and Megacu paths must use
  device-side NVSHMEM, not host-side reduction callbacks.
- The config capability range must be explicit: platform, backend, dispatcher,
  scheduler, lowering mode, supported team sizes, supported layouts/dtypes, and
  unsupported features.

## Verification Requirements

The PR is not complete until fresh evidence covers:

- compile-only authoring for phased and overlap descriptors;
- materialization tests for program facts and owned records;
- dispatcher tests that inspect tile/participant/backend peer placement;
- scheduler tests for phased readiness and co-resident overlap guard behavior;
- negative test for blocking communication without a valid residency guard;
- lowering tests for missing op symbols and no generated CUDA/C++ source;
- linked metadata tests for program, dispatch, schedule, kernel, and backend
  sections;
- direct ABI validation tests returning `megacu::status` before launch on bad
  runtime views;
- CUDA single-card numeric correctness for local GEMM golden, baseline phased,
  baseline overlap, Megacu phased, and Megacu overlap;
- single-host two-card CUDA/NVSHMEM correctness under Docker using device-side
  NVSHMEM for baseline and Megacu paths;
- file-organization checks for `examples/`, `docker/`, and `tools/`;
- `git diff --check`, CMake build, and CTest.

## Current Gap List

These gaps must be closed before the task can be marked complete:

- Replace the current golden naming with local-GEMM golden result generation,
  four pure CUDA+NVSHMEM baselines, and two Megacu implementations.
- Remove numeric validation from orchestrate/common runtime code; correctness
  checking belongs in tests or validation drivers.
- Ensure Megacu phased and overlap paths consume dispatcher/scheduler/lowering
  payloads rather than directly selecting golden/native helper calls.
- Keep local golden, CUDA+NVSHMEM baselines, and Megacu paths clearly
  separated.
- Keep examples in `<platform>_<backend>/<example>` trees. The
  CUDA+NVSHMEM GEMM+AllReduce example is one family root with
  `{common,golden,phased/{baseline,megacu},overlap/{baseline,megacu}}`
  underneath it. Keep shared platform/backend Docker and tool assets unless
  per-example support is justified.

## Landed Evidence

- `src/dispatcher/tiled_compute_comm_dispatch.cc` now owns tiled
  compute/communication dispatch metadata, including work entries and
  participant/backend peer mappings.
- `src/scheduler/static_persistent.cc` now owns phased and co-resident
  persistent schedule metadata, including schedule entries, event wait/release
  slots, residency groups, and CUDA residency envelope facts.
- `src/lowering/persistent_stitch.cc` now owns kernel symbol metadata and
  entrypoint-role records for the selected lowering mode.
- `src/platform/cuda/validation.cc` and
  `src/backends/nvshmem/validation.cc` now own reusable runtime validation for
  CUDA launch views, NVSHMEM teams, and symmetric storage.
- `cmake/MegacuTargets.cmake` now builds `cuda_nvshmem_static` from the
  component source files rather than from one `component_anchor.cc` placeholder.
- `tests/build/component_metadata_contracts.cc` verifies the dispatcher,
  scheduler, lowering, backend metadata, and linked component symbols.
- `tests/build/runtime_adapter_validation.cc` verifies reusable CUDA/NVSHMEM
  validation behavior.
- `tests/build/compile_gemm_allreduce_descriptor.cc` now consumes the example
  descriptor header instead of defining a duplicate orchestrate program inside
  `tests/`.
- `examples/cuda_nvshmem/gemm_allreduce/` now owns one example family layout:
  shared descriptor/runtime/golden code at the family root and phased/overlap
  variants split into `baseline/` and `megacu/` directories.
- `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/` and
  `examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/` now own separate
  example CMake files, READMEs, and orchestrate entrypoints.
- Shared Docker and tool support now live at `docker/cuda_nvshmem/` and
  `tools/cuda_nvshmem/`; the stale per-example Docker/tool support directories
  were removed.

Current local evidence:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
MEGACU_TEST_CUDA_DEVICES=5,6 MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build --output-on-failure
bash -n tools/cuda_nvshmem/run_two_card_docker.sh
test -d examples/cuda_nvshmem/gemm_allreduce/common
test -d examples/cuda_nvshmem/gemm_allreduce/phased/baseline
test -d examples/cuda_nvshmem/gemm_allreduce/phased/megacu
test -d examples/cuda_nvshmem/gemm_allreduce/overlap/baseline
test -d examples/cuda_nvshmem/gemm_allreduce/overlap/megacu
test -d examples/cuda_nvshmem/gemm_allreduce/golden
test ! -e examples/cuda_nvshmem/gemm_allreduce_phased
test ! -e examples/cuda_nvshmem/gemm_allreduce_overlap
test ! -e examples/cuda_nvshmem/common
git diff --check
```

Result: configure/build passed, 9/9 CTest tests passed, Docker script syntax
checked, requested GEMM+AllReduce layout checked, and `git diff --check`
passed locally on 2026-04-25 Asia/Shanghai.

## Tests To Run

Minimum local evidence after implementation changes:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
MEGACU_TEST_CUDA_DEVICES=5,6 MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build --output-on-failure
tools/cuda_nvshmem/run_two_card_docker.sh
git diff --check
```

If host NVSHMEM is unavailable, Docker-provisioned CUDA+NVSHMEM remains the
required two-card validation path. If Docker or GPUs are unavailable, record the
exact blocker and do not claim multi-card correctness.

## Closeout

Before this PR can be closed:

- update this task so completed items have evidence, not intent;
- update `docs/todo/README.md` for any remaining implementation gaps;
- keep accepted architecture content in `docs/design/` untouched until PR
  merge preparation;
- merge only implementation-relevant learnings into stable design docs at the
  end of the PR.
