# Verification

## Core Contracts

- The public authoring model is an orchestrate program, not a runtime-selected
  config object.
- The public runtime model is compiled orchestrate program -> `run(...)`.
- CMake/build does not appear in the runtime C++ API.
- Dispatcher, scheduler, kernel lowering, platform, and backend are selected by
  the build graph.
- Named kernels may use backend primitives inside the kernel body.
- Fine-grained overlap must remain possible without introducing a public
  fragment taxonomy.

## Failure Modes

- Runtime strategy selection: runtime C++ still chooses dispatcher, scheduler,
  kernel lowering, platform, or backend. This violates the build-graph strategy
  rule.
- Build leakage into runtime C++: runtime API exposes build/materialize work.
  This violates the language split.
- No artifact reuse: every concrete program recompiles dispatcher/scheduler/
  lowering engines instead of reusing compiled artifacts. This violates the
  CMake build-graph model.
- Plugin-shaped primary API: normal user code requires string path loading or a
  generic `runtime_env`. This violates the direct-compiled-runtime rule.
- Opaque-kernel trap: named ops cannot express fine-grained overlap because
  backend primitives are excluded from kernel bodies. This violates the overlap
  goal.
- Descriptor leakage: users are forced to author packed descriptors directly.
  This violates the program model.
- Mapping/scheduling conflation: dispatcher and scheduler are not clearly
  separated. This weakens implementation clarity.
- Distributed bootstrap gap: CUDA+NVSHMEM examples require a team handle but no
  adapter owns `torchrun`/`mpirun`, CUDA device selection, NVSHMEM bootstrap, or
  symmetric allocation. This makes the first multi-GPU proof impossible to run.
- Framework control-plane confusion: Torch Distributed collectives are used as
  the device communication backend instead of only as rank/world-size and UID
  exchange for NVSHMEM bootstrap.
- Team-size ambiguity: a target compiled for one team-size envelope is launched
  with a different NVSHMEM PE count.
- Unsafe overlap: a communication task performs a blocking wait while the
  producer task is not guaranteed to be co-resident and making progress. This can
  deadlock even if the logical event graph is correct.

## Verification Requirements

- CMake/build tests that produce reusable component targets
- CMake/build tests that produce phased and overlap orchestrate targets from
  those components
- build evidence that `PROGRAM gemm_allreduce_phased_program` and
  `PROGRAM gemm_allreduce_overlap_program` metadata are produced by the native
  C++ build path, not by Python or runtime parsing
- inspection evidence that `program_ir` contains the expected extents, domains,
  participants, resources, events, and submissions for GEMM+AllReduce
- inspection evidence that `dispatch_plan`, `schedule_plan`, `kernel_plan`, and
  `backend_plan` are materialized with the ownership described in
  `05-dispatcher-scheduler-kernel.md`
- design or compile evidence that each public concept in
  `01-program.md` is required by either scheduling, lowering, backend
  resolution, or runtime parameterization
- tests or inspection proving compiled function parameters fill pre-lowered
  slots and do not select dispatcher, scheduler, lowering, platform, or backend
  strategy
- native build checks that the artifact is produced by ordinary CUDA/C++
  compilation rather than runtime code generation
- checks proving kernel lowering and target lowering select/link existing
  implementations and materialize metadata rather than emit new C++/CUDA source
- checks proving `.megacu.bin` runtime metadata is a serializable data blob with
  no process-local C++ fields and is linked as data, not generated source
- checks proving linked metadata validation rejects bad magic, version, size,
  checksum, or missing required tables before any kernel launch
- checks proving CMake `OPS` entries resolve to the entrypoint roles required by
  the selected lowering mode, without requiring unused roles
- direct-call smoke tests for the compiled orchestrate program
- direct-call smoke tests proving compiled targets return `megacu::status` for
  validation, launch, and backend errors rather than throwing from the core ABI
- status ABI tests proving returned status messages point to static storage or
  null, not temporary strings
- linked-artifact or metadata inspection for the first CUDA/NVSHMEM target,
  including domain, participant, event, dispatch, schedule, and backend slots
- metadata inspection proving the phased target uses `progress_model::phased`
- metadata inspection proving the overlap target uses
  `progress_model::co_resident_persistent`, includes a residency group with both
  compute and communication workers, and rejects blocking waits without a valid
  guard
- repeated-run tests showing the internal `run(...)` path stays cheap
- integration tests showing CMake/build drive target creation while runtime C++
  only calls the compiled orchestration
- two-rank GEMM+AllReduce correctness tests for the first target where hardware
  is available
- MPI launch integration test or skip:
  `mpirun -np 2 ./cuda_nvshmem_gemm_allreduce_mpi ...`
- Torch launch integration test or skip:
  `torchrun --standalone --nnodes=1 --nproc-per-node=2 ...`
- tests or inspection proving `nvshmemx_init_attr`/`nvshmem_finalize` are owned
  by launch adapters or caller code, not by the compiled orchestrate target
- tests proving the Torch adapter validates Torch rank/world size against
  NVSHMEM PE id/count before kernel launch
- tests proving the MPI adapter validates MPI rank/world size against NVSHMEM PE
  id/count before kernel launch
- tests proving symmetric event and partial buffers are required when the
  backend plan requires remote NVSHMEM access
- tests proving symmetric event and partial buffers carry backend/session
  identities matching the launched `team_view`
- checks proving optional Torch integration stays outside Megacu core and does
  not introduce Python into public program, metadata, lowering, or runtime ABI

## Concrete Metadata Checks

The first metadata JSON must be checked for these facts:

- two runtime problem extents for output tiles: `m_tiles_extent` and
  `n_tiles_extent`;
- one backend-derived rank extent for `rank_domain`;
- one `output_tile_domain` over the two matrix tile extents;
- one `rank_domain` over the backend team-size extent;
- two virtual participants: `compute_lane` and `reduce_lane`;
- one workspace resource slot and one event-storage resource slot;
- one `partial_ready_event` over `(output_tile_domain, rank_domain)`;
- one `gemm_tile_produce` submission over `output_tile_domain` that releases
  `partial_ready_event`;
- one `allreduce_tile_consume` submission over `output_tile_domain` that
  acquires all rank instances of `partial_ready_event`;
- a dispatch plan with compute and communication placements;
- a schedule plan where GEMM production precedes AllReduce consumption through
  the event dependency, not a full-kernel implicit barrier;
- for the phased target: `progress_model::phased`;
- for the overlap target: `progress_model::co_resident_persistent` with one
  residency group containing at least one compute worker and one communication
  worker;
- for the overlap target: blocking acquires only wait on events produced by the
  same residency group or by a completed earlier phase;
- event-use metadata names `event_wait_mode` for each acquire, and any
  `blocking_device_wait` acquire is covered by the overlap guard;
- schedule entries include phase, residency group, participant role, and wait
  mode fields;
- the overlap `schedule_plan` includes a `cuda_residency_envelope` with
  persistent grid blocks, threads per block, minimum SM count, blocks per SM,
  and cooperative-launch requirement;
- a backend plan that names event storage size, event offsets, team size, and
  whether multimem reduce is enabled;
- a platform launch slot for `megacu::cuda::launch_view`;
- metadata header fields for magic, metadata ABI version, target id, component
  ids, table offsets/counts, and total size/checksum;
- op implementation metadata for each op slot naming only the entrypoint roles
  and symbol ids required by the selected lowering mode;
- a backend target envelope with `TEAM_SIZE 2` for the first proof;
- resource metadata marking `partial` and `events` as requiring symmetric
  NVSHMEM-accessible storage;
- runtime view metadata or adapter tests prove event and partial symmetric views
  carry the same backend/session identity as the launched `team_view`.

Any missing item means the design has not become implementation-ready.

## Example-To-Evidence Mapping

- Program API example in `01-program.md` and
  `07-first-validation-slice.md`:
  compile-only test under `tests/build/` plus direct-call smoke test.
- CMake examples in `02-cmake-build-and-runtime.md`:
  configure/build test proving component target reuse and orchestrate target
  linkage.
- Dispatcher, scheduler, and lowering contracts in
  `05-dispatcher-scheduler-kernel.md`:
  linked-artifact and metadata inspection showing op, resource, event,
  dispatch, participant, schedule, kernel, and backend payload ownership.
- Kernel context and backend primitive examples in `01-program.md` and
  `06-examples.md`:
  compile-only checks showing kernels use typed `kernel_context` APIs instead
  of string event lookup or hand-authored backend signal addresses.
- First slice runtime behavior in `07-first-validation-slice.md`:
  two-rank CUDA/NVSHMEM GEMM+AllReduce correctness tests through both MPI and
  Torch launch paths where hardware exists; explicit skip reason for each
  unavailable launcher or local NVSHMEM multi-GPU environment.
- Distributed launch contract in
  `09-distributed-launch-and-framework-integration.md`:
  adapter unit tests for team construction, device selection, symmetric
  allocation checks, and rank/world mismatch failures.
- Larger MPK-style example in `06-examples.md`:
  design/compile evidence that a serving-layer program can link CUDA-provided
  RMSNorm, linear, paged-attention, split-reduce, and residual-output operator
  bodies without changing the public program model.
- No runtime strategy selection:
  code inspection or test hook proving `run` does not call build, CMake,
  dispatcher selection, scheduler selection, backend selection, or string-based
  module loading.
- Implementation architecture guardrails in `10-implementation-architecture.md`:
  metadata ABI tests, dependency-boundary inspection, status ABI tests, and
  linked-op-symbol inspection.

## Ready-To-Promote Criteria

This active design is ready to merge back into `docs/design/` only when:

- every public surface has a planned owner and file path in the owning chapter;
- every example has a corresponding test, inspection, or skip rule;
- no stable doc points at unfinished draft content as implemented behavior;
- the first implementation slice can be built from the component contracts
  without introducing new public concepts.
