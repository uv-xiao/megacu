# Feature Task: Runtime-Linked Megacu Slice

- Branch: `implementation/public-builder-surface`
- PR: #3
- Owner: Codex
- Status: Active, architecture redirected

## Review Reset

PR #3 must no longer pursue the compiler-like implementation that records
program IR, materializes owned facts, builds dispatch/schedule/kernel/backend
sections, and validates linked target metadata. That model is too heavy and
does not match the intended Megacu layer.

The active design source is now
`docs/in_progress/design/implementation_ready_device_native_layer/`. The
previous stable copy was moved back from `docs/design/` because it is not
accepted implemented behavior.

## Corrected Design Contract

Megacu is a runtime-linked device-native layer:

- CMake links selected dispatcher, scheduler, platform, backend, target runtime,
  and native operator implementations.
- Public C++ exposes direct orchestrate ABI functions and typed runtime views.
- Dispatcher and scheduler are runtime components called inside the orchestrate
  target.
- Platform and backend adapters validate native CUDA/NVSHMEM handles and expose
  runtime/device-side primitives.
- Native CUDA/NVSHMEM operators execute the actual work.
- No normal implementation path uses `program_ir`, `owned_program_ir`,
  `materialize_program`, `target_metadata`, static `dispatch_section`, static
  `schedule_section`, or host materializer executables.

## Required File Organization

Core implementation ownership:

- `include/megacu/`: public status, runtime views, launch/team views, runtime
  config/context types.
- `include/megacu/detail/`: private runtime component declarations only.
- `src/dispatcher/`: runtime tile/rank/peer mapping.
- `src/scheduler/`: runtime phased and co-resident persistent scheduling.
- `src/platform/cuda/`: CUDA validation and launch helpers.
- `src/backends/nvshmem/`: NVSHMEM team, symmetric storage, and device-side
  primitive helpers.
- `src/target/`: direct target runtime glue and capability validation.
- `src/lowering/`: remove or rename unless a real runtime-linked operator
  ownership remains. Do not keep lowering as a compiler-stage name.

Example organization remains:

```text
examples/cuda_nvshmem/gemm_allreduce/
  common/
  golden/
  phased/baseline/
  phased/megacu/
  overlap/baseline/
  overlap/megacu/
```

Shared operational assets remain:

```text
docker/cuda_nvshmem/
tools/cuda_nvshmem/
```

## Working Slice Outputs

The PR should produce:

- Runtime public headers:
  - `include/megacu/views.h`
  - `include/megacu/platform/cuda.h`
  - `include/megacu/backends/nvshmem.h`
  - `include/megacu/runtime_config.h`
  - `include/megacu/runtime_context.h`
- Private runtime component declarations:
  - `include/megacu/detail/component_runtime.h`
- Runtime component implementations:
  - `src/dispatcher/*runtime*`
  - `src/scheduler/*runtime*`
  - `src/platform/cuda/*`
  - `src/backends/nvshmem/*`
  - `src/target/*`
- CMake functions that link runtime components and native operator symbols
  without generating source or metadata artifacts.
- Direct Megacu orchestrate targets:
  - `cuda_nvshmem_gemm_allreduce_phased_orchestrate`
  - `cuda_nvshmem_gemm_allreduce_overlap_orchestrate`
- Native paths:
  - local golden result generation;
  - pure CUDA/NVSHMEM phased baseline, single-card and two-card;
  - pure CUDA/NVSHMEM overlap baseline, single-card and two-card;
  - Megacu phased target, single-card and two-card through one runtime design;
  - Megacu overlap target, single-card and two-card through one runtime design.

## Runtime Component Requirements

### Dispatcher

- Runtime input: problem shape, tile shape, team view, capability envelope.
- Runtime output: compact dispatch state and tile work cursors.
- Must map logical ranks to backend peers from the runtime team.
- Must not infer roles by ad hoc scans over event records.
- Must not build heap-heavy static metadata tables.

### Scheduler

- Phased scheduler: run compute and communication according to tile readiness,
  avoiding false whole-program dependencies when tile readiness is available.
- Overlap scheduler: require a co-resident or persistent progress guard before
  allowing blocking communication waits.
- Scheduler decisions happen inside the orchestrate call, within the linked
  target capability envelope.

### Platform And Backend

- CUDA validates stream, device identity, launch shape, and persistent or
  cooperative constraints when required.
- NVSHMEM validates team size, PE identity, symmetric event storage, symmetric
  partial buffers, backend/session identity, and peer mapping.
- Multi-card communication must use device-side NVSHMEM in baseline and Megacu
  paths.

### Target Runtime And Operators

- Target runtime validates the linked capability envelope and runtime views.
- Native operator symbols are linked by CMake and called directly.
- No runtime strategy loading by string, path, or generated module.

## Distributed Requirements

Distributed coverage is part of this PR's design target:

- Single-process single-card must use the same direct orchestrate ABI with
  `team_n_pes == 1`.
- Single-host two-card must run through NVSHMEM, preferably via Docker when the
  host environment does not provide complete NVSHMEM.
- MPI launch support must be represented by an adapter that constructs the same
  `cuda::launch_view` and `nvshmem::team_view`; the orchestrate target must not
  depend on MPI headers.
- Torch-distributed support must be represented by a framework adapter that
  extracts tensor/stream/session views and calls the same direct orchestrate
  ABI; it must not reimplement dispatcher or scheduler decisions.
- Unsupported distributed cases must be named explicitly: arbitrary PE counts,
  multi-node performance claims, host-side collectives as Megacu
  communication, and dynamic scheduler queues.

## Verification Requirements

The PR is not complete until fresh evidence covers:

- no implementation dependency on `program_ir`, `materialize_program`,
  `owned_program_ir`, static `target_metadata`, static `dispatch_section`,
  static `schedule_section`, or static `kernel_section`;
- runtime dispatcher tests for tile/rank/peer mapping from concrete problem and
  team values;
- runtime scheduler tests for phased tile readiness and overlap progress guard;
- negative overlap test for blocking communication without a valid progress
  guard;
- CUDA platform validation tests;
- NVSHMEM backend validation tests;
- direct ABI validation tests returning `megacu::status` before launch on bad
  runtime views;
- CUDA single-card numeric correctness;
- two-card CUDA+NVSHMEM correctness under `nvshmrun` or Docker;
- MPI adapter smoke or documented blocker;
- torch-distributed adapter smoke or documented blocker;
- file-organization checks for `examples/`, `docker/`, and `tools/`;
- `git diff --check`, CMake build, and CTest.

## Current Gap List

- Remove compiler-like materialization implementation and tests.
- Replace static metadata builders with runtime dispatcher/scheduler/backend
  component calls.
- Replace ad-hoc dispatcher behavior with explicit GEMM+AllReduce runtime
  mapping.
- Move reusable target runtime validation out of example-only headers.
- Separate golden, baseline, and Megacu native paths in code.
- Add distributed runtime adapter design and evidence for NVSHMEM, MPI, and
  torch-distributed entry points.

## Tests To Run

Minimum evidence after implementation changes:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
MEGACU_TEST_CUDA_DEVICES=5,6 MEGACU_TEST_CUDA_DEVICE=6 \
  ctest --test-dir build --output-on-failure
bash -n tools/cuda_nvshmem/run_two_card_docker.sh
git diff --check
```

If Docker, MPI, torch, GPUs, or NVSHMEM are unavailable, record the exact
blocker and do not claim the corresponding distributed support.
