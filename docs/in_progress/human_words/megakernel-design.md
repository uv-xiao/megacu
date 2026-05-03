# Human Words: Megakernel Design

## Category

- Primary: Megakernel Design

## Historical Path Note

Older `Related:` paths below may point at `docs/in_progress/` locations that
were later promoted into `docs/design/`.

## Timeline

- 2026-04-22 Asia/Shanghai - Multi-GPU from the beginning
  > Multi-gpu from the beginning.
  - Context: User answered the first Megacu design-scope question about whether
    the first executable slice should target single-GPU static scheduling only
    or include multi-GPU/NVSHMEM concepts from the beginning.
  - Related: `docs/in_progress/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`
  - Agent interpretation: The initial architecture must model multi-GPU
    ownership, communication, and verification from day one rather than adding
    them as a later retrofit.

- 2026-04-22 Asia/Shanghai - Backend interface and source readings
  > You recommendation is good. But you should also clone triton-distributed and https://github.com/HazyResearch/Megakernels/tree/mk-v2-llama-70b (with the thunderkittens submodule init and updated) and https://github.com/microsoft/mscclpp. We need very careful design of the backend interface and how different backends get supported. Not only the interface, we need to do perfect architect and project organization to support the features in a modular, extensiable way to compose optimized CUDA kernels with minimal changes required.
  - Context: User accepted the recommendation for a backend interface with
    NVSHMEM first, then expanded the source-reading and architecture scope.
  - Related: `docs/in_progress/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`, future source note for
    distributed kernel backends.
  - Agent interpretation: The backend design must be informed by real
    distributed kernel systems and must emphasize project organization,
    modularity, extensibility, and low-friction composition of optimized CUDA
    kernels.

- 2026-04-22 Asia/Shanghai - Refine backend capability taxonomy first
  > 1
  - Context: User chose the first revision focus after being offered:
    "Backend capability taxonomy and device interface",
    "Project/module organization and ownership boundaries", or
    "First executable slice and verification plan".
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`
  - Agent interpretation: Revise Option B by making backend capability groups
    and device interface contracts more concrete before accepting the design.

- 2026-04-22 Asia/Shanghai - Do not rebuild transport libraries
  > make surr that we are not building new wheels. Instead, we should propose new abstraction to land techniques in a zero-overhead way. Especially, the abstragiton should brong all features of event-tensor to cuda, but dont write a nvshmem or mscclpp (instead, use them to impl). This principle or goal clarification is inportant.
  - Context: User clarified the goal after backend taxonomy refinement.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/megacu_cpp_cuda_layer.md`
  - Agent interpretation: Megacu should expose a CUDA-native abstraction for
    event-tensor-like programming and megakernel composition, while using
    NVSHMEM/MSCCL++/CUDA mechanisms as implementation backends instead of
    reimplementing communication runtimes.

- 2026-04-22 Asia/Shanghai - Avoid tight CUDA binding
  > i have new comments on this work. can we make it not tightly bound to cuda?
  - Context: User raised an architecture concern after the first executable
    slice and backend design were drafted around CUDA/NVSHMEM wording.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/megacu_cpp_cuda_layer.md`
  - Agent interpretation: The core abstraction should separate accelerator-
    and backend-neutral semantics from CUDA-specific platform bindings, while
    keeping CUDA as the first implementation and performance baseline.

- 2026-04-23 Asia/Shanghai - Positioning and design weight
  > Let's only keep the picked option in the document (B2: Platform-neutral Core With CUDA/NVSHMEM First). And I found that the design is kind of heavy: are things about core/memory needed to be provided? We shoudl discuss more about the design with rationale analysis. Before that, we need to make the work's position and story accurate. The several major competitors are: mpk, event-tensor, triton-dist (you can also download https://arxiv.org/pdf/2512.22219), and megakittens. What makes our work special and meaningful? Who are the audience?
  - Context: User asked to simplify the design doc and revisit the product
    story before committing to heavy module boundaries.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/notes/megakernel_cuda_layer_sources.md`,
    `docs/notes/distributed_backend_sources.md`
  - Agent interpretation: The design should lead with a clear competitor
    comparison, target audience, and rationale for why Megacu is a meaningful
    layer. Module names such as `core` and `memory` must be justified by the
    first executable slice rather than introduced as heavyweight architecture.

- 2026-04-23 Asia/Shanghai - Correct UniEP paper link
  > The link is wrong: https://arxiv.org/pdf/2604.19241v1
  - Context: User corrected the additional arXiv source to read for
    positioning against Triton-distributed/UniEP-style work.
  - Related: `docs/notes/distributed_backend_sources.md`
  - Agent interpretation: Use arXiv:2604.19241v1, UniEP, as the relevant
    additional paper. Treat the earlier arXiv:2512.22219 download as MPK
    background, not the corrected requested source.

- 2026-04-23 Asia/Shanghai - Structure design docs
  > Let's make the design docs/in_progress/design structured rather than monolithic, and we can then refine design of certain aspects one by one.
  - Context: User wants the accepted Megacu design split into focused documents
    so positioning, architecture, backend interfaces, examples, and evidence can
    be refined independently.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`
  - Agent interpretation: Keep the existing design path as a stable index, but
    move detailed design content into a subdirectory of focused aspect docs.

- 2026-04-23 Asia/Shanghai - Forward-looking shared concepts
  > During the design, we should reduce the slice concept. Our design should be Forward-looking. For example, we need to carefully think about if core and memory are truly sharable for different platforms, or what is the true shared concepts to be modeled by the megacu layer.
  - Context: User corrected the design process after the structured docs were
    created. The first executable slice should not over-determine the public
    abstraction.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Treat `cuda_nvshmem_event_copy` as validation evidence,
    not as the main source of abstraction. Model only concepts that are truly
    shared across platforms/backends, and keep `core`/`memory` as questioned
    candidates rather than assumed modules.

- 2026-04-23 Asia/Shanghai - Absorb shared concepts into platform interface
  > shared concepts should not be a standalone document, it should be absorted into platform-inferfact as rationale analysis or design evidence.
  - Context: User refined the structured design organization after the
    standalone shared-concepts chapter was added.
  - Related: `docs/in_progress/design/megacu_device_native_layer/05-language-responsibilities.md`,
    `docs/in_progress/design/megacu_device_native_layer/06-compiled-orchestrate-program.md`
  - Agent interpretation: Keep the shared-concept reasoning, but place it inside
    the platform-interface chapter as rationale/evidence instead of maintaining
    a separate chapter.

- 2026-04-23 Asia/Shanghai - Merge memory into shared and explain CUDA fusion
  > let's merge memory module into shared (remove `core`), and we must make it clear that one thing is modeled in shared only when it is the must-be dependency for event/task/schedule/kernel. This needs very careful analysis. For docs/in_progress/design/megacu_device_native_layer/examples.md, it should discuss how cuda kernels be fused with megacu's help.
  - Context: User refined the architecture boundaries and examples chapter.
  - Related: `docs/in_progress/design/megacu_device_native_layer/05-language-responsibilities.md`,
    `docs/in_progress/design/megacu_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/megacu_device_native_layer/08-examples.md`
  - Agent interpretation: Remove `core` as a module name, fold memory-model
    vocabulary into `shared`, and admit shared concepts only when event, task,
    schedule, or kernel contracts require them. Examples should explain how
    native CUDA kernel bodies are fused into Megacu tasks/schedules.

- 2026-04-23 Asia/Shanghai - Automatic iterative concretization
  > Now we need to make the design concrete.
  > We need to do this in an automatic, iterative way.
  >
  > For very iteration, you need to:
  > 1. Analyze if event/task/schedule can cover all features that related works (MPK, Event-tensor, triton-dist, MegaKittens) provide. If we exceed one, for example, avoiding the graph building overhead of MPK, we also succeed in the coverage.
  > 2. Enhance event/task/schedule abstraction/APIs according to analysis.
  > 3. Based on new event/task/schedule model, rethink and update the kernel module.
  > 4. Rethink what is necessary for the new event/task/schedule/kernel design, and update the shared module to only include the required dependencies.
  > 5. Update the platform/backend interface according to shared.
  >
  > Before start, do shared and platform/backend interface overlap? Since we need the megacu layer thin, we cannot tolerate redundant concepts.
  - Context: User defined the process for making the design concrete and asked
    whether shared and platform/backend interfaces overlap.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Add a repeatable iteration protocol and a strict
    no-overlap rule: `shared` owns semantic dependency vocabulary required by
    event/task/schedule/kernel; platform/backend own native lowering, resource
    handles, and capability evidence.

- 2026-04-23 Asia/Shanghai - Tiny but mighty iteration expectation
  > We should add a strong requirement or expection for the iteration: we want the layer tiny but mighty, and we need to achieve the desired features with the minimal design.
  - Context: User refined the iterative design protocol while iteration 1 was
    being added.
  - Related: `docs/in_progress/design/megacu_device_native_layer/91-redesign-protocol.md`
  - Agent interpretation: Every iteration must minimize API surface while still
    covering or exceeding related-work features. New concepts require explicit
    coverage, cost, and must-be-dependency rationale.

- 2026-04-23 Asia/Shanghai - Logic-rich ordering and iteration summaries
  > The ordering should be logic-rich. And modifications should be applied into each document, with the changes summarized in iteration-k documents.
  - Context: User refined how structured design docs should be browsed and how
    iteration records should be used.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Rename chapters so browsing follows design logic, not
    arbitrary chronology. Apply canonical design changes directly to the relevant
    chapter files; use iteration records only as summaries and rationale logs.

- 2026-04-23 Asia/Shanghai - Framework-friendly minimal layer
  > Add a new requirement: the minimal megacu layer must be friendly for high-level framework integration, such as pytorch, sglang, and etc. You can clone and look at flashinfer to see how it enables the integration (somehow cudagraph support, but we don't need cudagraph, so our integration might be other aspects).
  - Context: User added an integration requirement after the first concrete
    Megacu design iteration and asked for FlashInfer source reading as evidence.
  - Related: `docs/notes/framework_integration_sources.md`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Megacu must stay thin and device-native while exposing
    stable descriptor, workspace, plan/run, optional-output, diagnostics, and
    packaging hooks that make PyTorch, SGLang, vLLM-style serving, and similar
    frameworks able to integrate it without rewriting Megacu or paying hidden
    hot-path overhead.

- 2026-04-23 Asia/Shanghai - Three minimization iterations
  > Run another 3 iterations to further improve the design, the goal is to minimize the megacu layer.
  - Context: User asked for three additional automatic design-refinement
    iterations after the framework integration requirement was added.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Run three canonical design passes that preserve
    related-work and framework coverage while deleting, demoting, or relocating
    concepts that are not must-be dependencies of event/task/schedule/kernel.

- 2026-04-23 Asia/Shanghai - More minimization iterations
  > Run more iterations to find things to improve.
  - Context: User asked to continue the automatic refinement loop after
    iterations 3-5 minimized launch resources, event/task/schedule, and the
    shared/backend boundary.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Continue searching for removable concepts and sharper
    boundaries. Treat new iteration records as improvement logs, while applying
    canonical design changes directly to the topic chapters.

- 2026-04-23 Asia/Shanghai - Redesign around thinner core and configuration
  > "Static,
  > inspectable schedules are the default. Dynamic queues, scheduler CTAs, or graph
  > builders are explicit features with explicit cost." About this, we don't need the core abstraction to provide multiple options at the same time (like different APIs). Instead, we need to keep interface/APIs very thin and concise, and we should use different configurations (like deciding what to link? I'm not sure about implementation detailss) to pick the expected scheduler or other features. Besides: Usage problems:
  > 1. Will tasks be explicited constructed by setting fields of the task_traits? Or they are collected by APIs.
  > 2. What are the descriptors? How they are built?
  >
  > Check redundancies:
  >
  > 1. task_kind and task_desc
  > 2. what is task regions, reads, writes
  > 3. is task_ctx used?
  > 4. static schedule and dynamic schedule should share a base class. They should be derived configurations, not natively different structs and paths.
  > 5. Although we give different related works, we don't need compatibility for all. We need our own solution, not copying all their mechanism to achieve coverage.
  > 6. There are so many `id`s. We need to keep them simple in core API, and provide their mapping to true rank/worker/... in a specific configuration.
  > 7. not understand what the coodinates and dimensions do.
  > 8. The kernel shape is also part of a specific configuration. We need to carefully consider what is core API/inferface and what is configuration and how they connect. I think configuration includes platform and backend registry.
  > 9. We also need to discuss plan-run model with core abstraction and configuration in mind.
  > 10. Look at examples, I found that we need to program complex tasks with Cpp templates. You can look at pto-runtime (clone https://github.com/hw-native-sys/simpler), where example orchestrator is programmed without such complexity (https://github.com/hw-native-sys/simpler/blob/main/examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp).
  > . This needs a careful redesign to solve the problems. You need to clean existing iteration notes and do the redesign with careful thinking
  - Context: User reviewed the current minimized design and concluded that the
    remaining API shape is still too heavy and too exposed. They want a
    redesign that separates a tiny core interface from configurable scheduler,
    kernel-shape, platform, and backend choices, and they want the examples to
    reflect a simpler orchestration style.
  - Related: `docs/in_progress/design/megacu_device_native_layer/`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/megacu_cpp_cuda_layer.md`
  - Agent interpretation: Rework the canonical design around a thinner core API
    and explicit configuration boundary, remove redundant concepts, clarify how
    tasks/descriptors are built, simplify ids/coordinates, revisit plan/run,
    use the `simpler` orchestration example as evidence, and clean iteration
    notes so they reflect the redesigned structure rather than the superseded
    incremental path.

- 2026-04-23 Asia/Shanghai - Reorganize design files for clarity
  > You don't need to keep the current file organization. You can reorganize files to make the re-designed solution logically fluent and clear.
  - Context: User reviewed the redesigned content and asked for the file layout
    itself to be changed so the browsing order matches the new thin-core plus
    configuration model.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: The chapter set and directory layout may be changed.
    Prior file numbering and chapter grouping are no longer constraints; favor a
    structure that makes the redesigned solution read cleanly from concept to
    configuration to validation.

- 2026-04-24 Asia/Shanghai - Prefer `orchestrate` over `bind`
  > I don't like the bind term, it implies too few things. Maybe orchestrate is better.
  - Context: User reviewed the runtime-stage naming in the redesigned flow and
    asked for a stronger term than `bind`.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: The runtime instantiation step should be renamed away
    from `bind`. The naming model should be reworked so `orchestrate` can be
    used cleanly without colliding with the authored program concept.

- 2026-04-24 Asia/Shanghai - Build should not appear in C++ runtime API
  > You should do a thorough redesign along the way and with the naming model. Also, you need to carefully think the compile-execute flow, especially in what language do what tasks. This is key for implementation. Especially, I think (not sure if correct or not) build/compile things cannot appear in Cpp.
  - Context: User approved another redesign pass and highlighted compile/execute
    flow and language responsibility as the key implementation question.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/`
  - Agent interpretation: Redesign the architecture around a clearer
    build/materialize versus runtime boundary, explicitly assign tasks to
    Python/tooling versus C++/CUDA/runtime, and avoid exposing build/compile as
    part of the C++ runtime API.

- 2026-04-24 Asia/Shanghai - Flatten and order the design files
  > Make the design document files flatten and ordered.
  - Context: User reviewed the redesign tree and asked for a flatter browsing
    structure with explicit ordering.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/00-overview.md`
  - Agent interpretation: Keep the redesign in a single ordered directory
    rather than nested subdirectories. Chapter names and numbering should make
    the reading path explicit.

- 2026-04-24 Asia/Shanghai - Megacu should be C++ only
  > Let's make megacu only Cpp, no python.
  - Context: User tightened the implementation boundary after the compile/execute
    flow was redesigned.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/megacu_device_native_layer/05-language-responsibilities.md`
  - Agent interpretation: Remove Python from Megacu's design story. Program
    authoring, build flow, and packaging should all be expressible through C++
    sources plus native build rules, while keeping build out of the runtime
    C++ API.

- 2026-04-24 Asia/Shanghai - Split reusable compilation from later target generation
  > I don't think the build should compile the program. Is it better to only compile things like dispatcher, while keep orchestrator not compiled until before time to run. This is a two-step compilation which can reuse compiled artifacts in the first step .
  - Context: User refined the C++-only build/runtime story after Python was
    removed.
  - Related: `docs/in_progress/design/megacu_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/megacu_device_native_layer/07-dispatcher-scheduler-kernel.md`
  - Agent interpretation: Separate reusable component compilation from later
    program-target generation. Dispatcher/scheduler/kernel-lowering engines
    should be compiled once and reused; the concrete program should reuse those
    artifacts through the build graph, without moving hidden compilation into
    `run`.

- 2026-04-24 Asia/Shanghai - Prefer CMake-managed linked runtime over strings and env bags
  > I dislike things like assemble. And I cannot accept things like auto module = megacu::load_execution_module("paged_attention.cuda_nvshmem.so");
  >
  > runtime_env env;
  > env.resource(resources::workspace, workspace_ptr, workspace_bytes);
  > env.resource(resources::events, event_buffer, event_bytes);
  > env.backend_handle(nvshmem_comm);
  >
  > auto exec = module.orchestrate(env); where we load things based on string reference and I don't know why we need the env. Why not just use Cmake to manage the build step, while keeping the orchestrate program clean (just call APIs which loads .so according to Cmake).
  - Context: User rejected the plugin-style runtime surface after the two-step
    build idea was introduced.
  - Related: `docs/in_progress/design/megacu_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/megacu_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/megacu_device_native_layer/08-examples.md`
  - Agent interpretation: CMake should own the primary build graph. The normal
    runtime path should compile and call the authored orchestrate program
    directly rather than use string-based module loading plus a generic
    `runtime_env`. Any plugin-style ABI should be optional and not the main
    design story.

- 2026-04-24 Asia/Shanghai - Flatten and order the design files
  > Make the design document files flatten and ordered.
  - Context: User reviewed the redesign tree and asked for a flatter browsing
    structure with explicit ordering.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/00-overview.md`
  - Agent interpretation: Keep the redesign in a single ordered directory
    rather than nested subdirectories. Chapter names and numbering should make
    the reading path explicit.

- 2026-04-24 Asia/Shanghai - Megacu should be C++ only
  > Let's make megacu only Cpp, no python.
  - Context: User tightened the implementation boundary after the compile/execute
    flow was redesigned.
  - Related: `docs/in_progress/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/megacu_device_native_layer/05-language-responsibilities.md`
  - Agent interpretation: Remove Python from Megacu's design story. Program
    authoring, build/materialize flow, and packaging should all be expressible
    through C++ sources plus native build rules or native offline tools, while
    keeping build out of the runtime C++ API.

- 2026-04-24 Asia/Shanghai - Split reusable compilation from program assembly
  > I don't think the build should compile the program. Is it better to only compile things like dispatcher, while keep orchestrator not compiled until before time to run. This is a two-step compilation which can reuse compiled artifacts in the first step .
  - Context: User refined the C++-only build/runtime story after Python was
    removed.
  - Related: `docs/in_progress/design/megacu_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/megacu_device_native_layer/07-dispatcher-scheduler-kernel.md`
  - Agent interpretation: Separate offline reusable component compilation from
    later orchestrate-target compilation. Dispatcher/scheduler/kernel-lowering
    engines should be compiled once and reused; the concrete program should
    reuse those artifacts through the build graph, without moving hidden
    compilation into `run`.

- 2026-04-24 Asia/Shanghai - Start implementation-ready design PR
  > Let's start a new PR to do implementation-ready design. The first step is to only keep the picked things in the design document docs/design, and remove those stale ones. Then we need to make things concrete in docs/in_progress/design. While we work in docs/in_progress/design, the docs/design should just be kept untouched, and we'll do document merging at the end of the PR.
  - Context: User requested a new PR-scoped design workflow before continuing
    concrete implementation-ready architecture work.
  - Related: `docs/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/architecture/`,
    `docs/in_progress/implementation_ready_device_native_design.md`
  - Agent interpretation: Keep `docs/design/` as a concise picked-direction
    snapshot, remove unfinished detailed chapters from stable design docs, do
    concrete refinement only under `docs/in_progress/design/`, and merge
    accepted documents back into `docs/design/` at PR closeout.

- 2026-04-24 Asia/Shanghai - Component-owned implementation contracts
  > We shouldn't put every thing in 11-implemented-contracts.md. Instead, we need to make very component of the design concrete, including the core abstraction APIs (event, task, ...) and all needs to be implemented.
  - Context: User reviewed the first PR slice and rejected concentrating
    implementation detail in one catch-all contract chapter.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Move concrete APIs, planned paths, records, and
    evidence into the owning component chapters. The active design should make
    each implementable component concrete instead of centralizing all detail in
    a final implementation-contract appendix.

- 2026-04-24 Asia/Shanghai - Explain orchestrator, operators, execution, and terms
  > I found things /APIs are not concrete enough. Fundamentally, we need to make it clear what we program in orchestrator and kernels (operators), and how they will be runned. For example, when we write event (with a name) or call primitives in a kernel for communication with another kernel (identified by like a virtual id?), how they will be runned with backend things provided (like providing how virtual id is resolved and event names are resolved). Also, all implementation APIs are quite opaque still. Another problem is that we don't give enough explanation for concepts/terms yet. For example, I just cannot understand what the workspace or domain mean.
  - Context: User reviewed the component-owned contracts and found that the
    design still lacked a concrete execution model and term definitions.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Define the vocabulary before APIs, distinguish
    diagnostic labels from typed tags, specify what users program in the
    orchestrator versus kernels, and document how lowering resolves domains,
    virtual participants, event tags, workspace views, and backend handles into
    runnable CUDA/NVSHMEM behavior.

- 2026-04-24 Asia/Shanghai - Clarify tiles, tile, workspace, events, bind, and extent
  > You should make it more cler about that is tiles, tile, workspace, events, and what exec.bind , megacu::extent are.
  - Context: User reviewed the descriptor/executor model and called out
    specific opaque terms in the first API example.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/04-compiled-orchestrate-program.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`
  - Agent interpretation: Explain `tiles` as runtime extent count, `tile` as
    logical domain, `workspace` as caller-owned payload/scratch storage,
    `events` as caller-owned synchronization storage, `exec.bind` as typed slot
    binding, and `megacu::extent` as typed runtime extent binding.

- 2026-04-24 Asia/Shanghai - Require necessity analysis
  > are these concepts and binding behavior necessary? Give rationale or necessity analysis.
  - Context: User asked whether the newly clarified concepts and runtime
    binding behavior are justified or removable.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Add explicit necessity analysis for each public
    concept and compare typed runtime binding against generic env bags,
    positional arguments, rebuild-per-shape, raw kernel plumbing, and
    Megacu-owned allocation.

- 2026-04-24 Asia/Shanghai - Dispatcher owns virtual participant mapping
  > This is weird. Why config in CMake? Why not do in dispatcher?
  - Context: User pointed at `MAP producer_lane TO RANK 0` /
    `MAP consumer_lane TO RANK 1` in the active design and questioned why
    participant placement was represented as CMake configuration.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/06-examples.md`
  - Agent interpretation: Virtual participants are logical role declarations.
    CMake may select the dispatcher component, but dispatcher policy owns
    participant-to-backend placement and emits participant mapping metadata.

- 2026-04-24 Asia/Shanghai - Parameterized compiled orchestrate function
  > I found the split between orchestrate-target structure and run-time dynamics not very necessary. Why not made orchestrate just to be a parameterized one? Why we still need exec.bind and exec.run upon compiled orch?
  - Context: User questioned the public `executor` / `bind` / `run` layer after
    the target has already been compiled.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/04-compiled-orchestrate-program.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Keep descriptor slots as lowering internals, but make
    the public compiled target a normal parameterized orchestrate function.
    Runtime parameter binding and fast-path `run` are generated/linked target
    internals, not public post-compile APIs.

- 2026-04-24 Asia/Shanghai - Lowering links existing implementations
  > We need to make sure things like kernel-lowering and target-lowering doesn't emit/generate new code. Instead, it mainly link the required low-level implementation to the high-level APIs as the backend provides.
  > We need to specify it very clear about what megacu's compilation do (compile, link) and not do (translation into new CUDA or other language code).
  - Context: User clarified that target/kernel lowering should not be a source
    generation or translation pipeline.
  - Related: `docs/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/03-language-responsibilities.md`,
    `docs/in_progress/design/architecture/04-compiled-orchestrate-program.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Megacu compilation means native compile/link plus
    compact metadata materialization. Kernel lowering and target lowering must
    select and link reusable dispatcher, scheduler, lowering, platform,
    backend, and kernel implementations; they must not emit new C++/CUDA or
    translate the orchestrate program into another language.

- 2026-04-24 Asia/Shanghai - Strengthen examples with GEMM+AllReduce and MPK
  > The example is too weak. We need a gemm-allreduce fusion-kernel as the example. You can get information and reference code from related repositories. Also, we need another bigger example from mpk (where cuda is provided).
  - Context: User reviewed the event-copy example and found it insufficient for
    implementation-ready design.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/04-compiled-orchestrate-program.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`,
    `docs/notes/distributed_backend_sources.md`,
    `docs/notes/megakernel_cuda_layer_sources.md`
  - Agent interpretation: Replace the toy event-copy proof with a
    GEMM+AllReduce fusion target that exercises real compute/communication
    overlap, then add an MPK-style serving-layer example grounded in MPK's
    CUDA-provided operator families.

- 2026-04-24 Asia/Shanghai - Design still not concrete enough
  > I don't think the design is concrete enough to guide implementation. We need to refine things further.
  - Context: User reviewed the GEMM+AllReduce and MPK example update and found
    that the design still lacked enough implementation detail.
  - Related: `docs/in_progress/design/architecture/00-overview.md`,
    `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/04-compiled-orchestrate-program.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: Implementation-ready design must include concrete
    records, owner paths, slot binding, materialization flow, function-body
    shape, and metadata checks, not only concept explanations and examples.

- 2026-04-24 Asia/Shanghai - Restrict MPK source evidence
  > For example creation, we need to make it clear that not everything inside mpk repo relates to mpk, since the original mirage things are outdated but still kept in the repo. Only things in research/repos/mirage-mpk/src/kernel and research/repos/mirage-mpk/python/mirage/mpk should be thought about as mpk-related.
  - Context: User corrected the source boundary for MPK-based examples and
    design evidence.
  - Related: `docs/in_progress/design/architecture/06-examples.md`,
    `docs/notes/megakernel_cuda_layer_sources.md`
  - Agent interpretation: Treat only `src/kernel/` and `python/mirage/mpk/`
    under `research/repos/mirage-mpk/` as MPK evidence for active examples.
    Other Mirage repository paths may be historical context, but should not be
    used as MPK-related support for design decisions without a new explicit
    source-reading justification.

- 2026-04-24 Asia/Shanghai - Fill multi-GPU running gap
  > review the implementation documents to see if it is complete or concrete enough. What I found not enough is about the multi-gpu running. We need to support CUDA+NVSHMEM, but I don't see where megacu is integrated for torchdistributed running or mpi running.
  >
  > We need to fill the missing thigns.
  - Context: User accepted the review finding that the implementation-ready docs
    were missing the CUDA+NVSHMEM process-launch and framework-integration
    contract.
  - Related: `docs/in_progress/design/architecture/09-distributed-launch-and-framework-integration.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`,
    `docs/notes/distributed_launch_sources.md`
  - Agent interpretation: The design must specify how Torch Distributed and MPI
    process models construct CUDA launch views, NVSHMEM team views, symmetric
    allocations, and validation evidence before the compiled orchestrate target
    runs.

- 2026-04-24 Asia/Shanghai - Separate phased and overlap GEMM+AllReduce
  > Another problem is that, should we support two version of gemm-ar? One is just like MPK, where, in my mind, omit compute-comm overlap. The other one is the overlapping version. Especially for the overlapping one, I'm worried that we don't have the guard mechanism for two communicating tasks must run simultaneously when the communication happens.
  - Context: User identified that the GEMM+AllReduce example was mixing a simple
    correctness baseline with a true compute/communication overlap proof, and
    noted the deadlock risk when communicating tasks are not simultaneously
    resident.
  - Related: `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/07-first-validation-slice.md`,
    `docs/in_progress/design/architecture/08-verification.md`
  - Agent interpretation: The design should support both phased and overlapping
    GEMM+AllReduce targets. Blocking communication in the overlap target requires
    scheduler/kernel-lowering metadata that proves producer and consumer workers
    are co-resident, or target lowering must reject the program.

- 2026-04-24 Asia/Shanghai - Keep in-progress docs implementation-only
  > docs/in_progress/design/implementation_ready_device_native_layer hold may implementation-unrelated information, some files and some contents inside files. We should leave these architecture things or decisions in docs/design (already there), and let docs/in_progress only hold implementation-related things. This should make the in_progress documents more concise.
  - Context: User reviewed the implementation-ready draft after concrete launch
    and overlap contracts were added and found that stable architecture
    narrative was still mixed into the active implementation docs.
  - Related: `docs/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/architecture/`
  - Agent interpretation: Keep architecture positioning, principles, and stable
    decisions in `docs/design/`. The active implementation-ready draft should
    retain only implementation surfaces, records, owner paths, runtime paths,
    examples, failure checks, and verification evidence.

- 2026-04-24 Asia/Shanghai - Re-review implementation design for architecture health
  > Review and check if the documents are complete and concrete enough, and also check if it meets the high-level architecture requirements or story. Also, you need to do good architecting again to make sure the implementation is robust, healthy, maintainable.
  - Context: User asked for another architecture-quality pass after the active
    implementation docs were made more concise.
  - Related: `docs/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/architecture/`
  - Agent interpretation: Re-check completeness against the high-level lifecycle
    and C++-only story, then tighten implementation contracts that would affect
    robustness, maintainability, metadata safety, runtime ABI stability, and
    component ownership.

- 2026-04-24 Asia/Shanghai - Do not force host/device op symbol pairs
  > Why we need both host_symbol and device_symbol. Although for simpler (pto runtime), host (orch) and device (chip) is separated, in CUDA, we prefer to have everything as kernel. That is, we don't always have host. Will the explicit seperation of host_symbol and device_symbol harm the generalithy?
  - Context: User reviewed the op implementation ABI added during the
    architecture-hardening pass and questioned whether explicit host/device
    symbol fields overfit PTO Runtime or harm CUDA generality.
  - Related: `docs/in_progress/design/architecture/01-program.md`,
    `docs/in_progress/design/architecture/02-cmake-build-and-runtime.md`,
    `docs/in_progress/design/architecture/05-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/architecture/06-examples.md`,
    `docs/in_progress/design/architecture/10-implementation-architecture.md`
  - Agent interpretation: Model op implementation entrypoints by capabilities
    required by the selected lowering mode, not mandatory host/device pairs.
    A CUDA `__global__` launchable kernel is a valid op implementation without a
    separate device-callable body; a callable body is only required when a
    stitched lowering needs to call the op from within another kernel.

- 2026-04-24 Asia/Shanghai - Keep Megacu thinner than related systems
  > I looked through the design, and felt that the design is kind of heavy: there are lots of plans, records, views, ... I'm not sure if each is necessary, you should do another alignment to make sure our megacu layer is kept thin. Actually, I think mpk, triton-dist, megakittens are all quite thin. We should be thiner rather than heavier, even we have multi-platform/backend quirements.
  - Context: User reviewed the implementation-ready draft after API and
    metadata contracts were made concrete and found the design presentation
    still too heavy.
  - Related: `docs/design/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/architecture/`
  - Agent interpretation: Keep the public Megacu layer thinner than related
    systems. Plans, records, and views must be minimized; implementation plans
    should be private target-metadata sections rather than extra public
    abstractions. Multi-platform/backend support should come from narrow
    component adapters and build-time selection, not from a broad public
    lifecycle or taxonomy.

- 2026-04-24 Asia/Shanghai - Implement complete Megacu in one PR
  > We shouldn't only implement a single slice in this PR, since we cannot verify. You need to implement the complete megacu in this PR.
  - Context: User reviewed the first implementation PR after it only added the
    public compile-only builder surface.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: PR #3 must broaden from a narrow authoring-surface
    slice to a locally verifiable first Megacu implementation covering the full
    design lifecycle: authoring, materialization, metadata sections, CMake
    target plumbing, direct orchestrate ABI, validation, and negative checks.
    Hardware-dependent CUDA+NVSHMEM execution may still be skipped with an
    explicit reason when unavailable, but the implementation cannot stop at a
    non-verifiable public API slice.

- 2026-04-24 Asia/Shanghai - Validate available CUDA hardware before NVSHMEM Docker
  > This server has CUDA runtime, why we cannot validate on real device?

  > We can first do single-device multi-card validation. And then we can use docker to setup nvshmem and run it on our single host, two-card.
  - Context: User corrected the first implementation PR verification strategy
    after local inspection showed CUDA runtime, multiple A100 GPUs, Open MPI,
    and Docker were available, while host NVSHMEM tooling was not installed.
  - Related: `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: The implementation PR should not skip hardware
    validation just because host NVSHMEM is missing. It should first validate
    CUDA on real devices, including a single-host two-card CUDA smoke path, and
    then add Docker-provisioned CUDA+NVSHMEM validation for a two-card single
    host run.

- 2026-04-24 Asia/Shanghai - Require correctness before closing risks
  > For questions: 1. until real correctness test; 2. implement a numeric path. You should move on to solve the risks and missing things.
  - Context: User answered the open review questions about how far PR #3
    should go before it can be considered implementation-ready.
  - Related: `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: Smoke tests are not enough for this PR. The first
    implementation must include a numeric GEMM+AllReduce path and keep closing
    review risks until correctness is validated on real CUDA and CUDA+NVSHMEM
    paths, even if the first numeric path is intentionally small.

- 2026-04-25 Asia/Shanghai - Require golden phased and overlap examples
  > We should provide two golden: golden_phased, golden_overlap. Also, I didn't see a persistent loop for gemm_allreduce, which is required for overlapping verison. You can look at research/repos/triton-distributed/python/triton_dist/kernels/nvidia/gemm_allreduce.py, and record how the fused kernel is implemented with triton-dist. For the phased one, can we also do some tiling for  some task-level overlapping?
  - Context: User reviewed the CUDA+NVSHMEM GEMM+AllReduce example after the
    first numeric path and example organization landed in PR #3.
  - Related: `examples/cuda_nvshmem/gemm_allreduce/`,
    `docs/notes/distributed_backend_sources.md`, PR #3.
  - Agent interpretation: The example should grow from one minimal numeric path
    into two named golden examples. The overlap golden needs a persistent
    compute/communication loop resembling the Triton-Distributed fused kernel.
    The phased golden should also use tiled work and readiness so it can show
    task-level overlap boundaries, even if communication remains phase-gated.

- 2026-04-25 Asia/Shanghai - Golden baselines separate from two Megacu implementations
  > Just remove the minimal numeric path. For golden mode, we just use pure cuda/nvshmem without megacu things to achieve the correct version. For each, it should support two version (one card and multiple cards).  And then, we need to use megacu to implement the two corresponding versions. Especially, for the phased one, the megacu version should be able to make gemm tile and ar tile happen together (loosing false dependencies) whenever data is ready. I think megacu should be able to run the design on single card or multiple cards automatically (depends on dispatcher, backend, ...) when the CMake configured. Another requirement for agent rules is to let each example has its own cmake file, not using the one at the repo root.

  > "Then add four corresponding Megacu implementations with the same semantics" I think we only need two Megacu implementation, since megacu should be able to generalize from single card to multiple cards.
  - Context: User refined the approved golden example direction before
    implementation.
  - Related: `examples/cuda_nvshmem/gemm_allreduce/`,
    `.agents/rules/example-organization.md`, PR #3.
  - Agent interpretation: Keep golden baselines Megacu-free and platform
    native. Provide single-card and multi-card golden variants for both phased
    and overlap, but only two Megacu implementations: phased and overlap. The
    Megacu implementation should choose single-card versus multi-card behavior
    from the target/team/backend configuration rather than duplicating program
    code.

- 2026-04-25 Asia/Shanghai - Require device-side NVSHMEM and config capability range
  > we need the device-side one in both golden and our megacu solution. Also, we need to be very clear about one megacu config (including platform, backend, dispatcher, schediler, etc.)'s capability range, especially for our goal to share one design for single-/multi-card scenario.
  - Context: User reviewed the two-rank GEMM+AllReduce path and rejected
    host-side NVSHMEM reduction callbacks as the multi-card communication
    mechanism.
  - Related: `examples/cuda_nvshmem/gemm_allreduce/`,
    `tests/runtime/nvshmem_two_rank_smoke.cc`, PR #3.
  - Agent interpretation: The golden CUDA+NVSHMEM path and the Megacu path must
    perform multi-card communication with NVSHMEM APIs called from device code.
    The example must also state the concrete capability envelope of the
    `cuda_nvshmem_static` config, including what single-card and multi-card
    cases are shared by one design and what is intentionally out of scope.

- 2026-04-25 Asia/Shanghai - PR #3 is not yet a working slice
  > I've glanced over the PR. You should read my comments. I think there are large problems about feature implementation (don't see any dispatch/schedule/etc., don't think we've implemented a working version) and file organization. You should look at research/repos/simpler/.claude/skills/fix-pr/ to $skill-creator create a skill about get and fix PR comments, and then rethink about docs/in_progress/public_builder_surface.md, making it a full working slice (aligned with the complete implementation-ready design).
  - Context: User reviewed PR #3 after numeric CUDA/NVSHMEM examples were added
    and found that the implementation still did not demonstrate the complete
    Megacu lifecycle from the accepted implementation-ready design.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: The active PR task must be reset around a real working
    Megacu slice. Metadata labels and example-local wrappers are not enough;
    the task must explicitly require implemented dispatcher, scheduler,
    lowering, CUDA platform, NVSHMEM backend, target-lowering, file
    organization, and verification paths aligned with the accepted design.

- 2026-04-25 Asia/Shanghai - PR #3 pending review details
  > The overlap golden seems fake.
  >
  > I think we are doing wrong. We need:
  > 1. golden is just local GEMM to give golden results;
  > 2. baseline* should be pure CUDA+NVSHMEM implementation of GEMM-AR; -- this needs four versions: single/multiple cards, phased or overlapped.
  > 3. megacu* should be Megacu-based implementation. -- this needs two versions.
  >
  > And we need to put phased and overlapped as two examples, don't put in the same directory.
  >
  > numeric validation should be removed.
  >
  > this looks very fake. Where is the working implementation?
  >
  > Why we put the orch code inside tests/ programs?
  >
  > The implementation is so thin. Where is the components? This looks not working at all.
  - Context: These comments were discovered under a pending GitHub review on PR
    #3 after normal submitted-comment endpoints returned no comments.
  - Related: `docs/in_progress/runtime_linked_megacu_slice.md`,
    `examples/cuda_nvshmem/gemm_allreduce/`, `tests/build/`, `src/components/`.
  - Agent interpretation: Split GEMM+AllReduce phased and overlap into separate
    examples, distinguish local-GEMM golden results from pure CUDA+NVSHMEM
    baseline implementations and Megacu implementations, remove correctness
    validation from orchestrate/common runtime code, move orchestrate program
    definitions out of tests, and implement real Megacu components instead of a
    placeholder component anchor.

- 2026-04-25 Asia/Shanghai - Document the concrete implementation path
  > I found the implementation is quite thin and not well-documented. Let's create documents for implementation in docs/in_progress/design/concrete_impl/, which holds the explain for files under include/ and src/. And we also need the path visulaization and explanation to describe how a megacu-based example can run with the impls: what functions are called one-by-one.
  - Context: User reviewed the implementation after the initial component
    pipeline and example reorganization landed in PR #3.
  - Related: `docs/in_progress/design/concrete_impl/`, `include/megacu/`,
    `src/`, `examples/cuda_nvshmem/gemm_allreduce/`.
  - Agent interpretation: Add implementation-facing documentation that maps
    current files to responsibilities and traces a Megacu-based example from
    authored descriptor through build/linking, metadata validation, runtime
    validation, and native CUDA/NVSHMEM execution. The docs must be honest
    about current thinness instead of overclaiming completeness.

- 2026-04-25 Asia/Shanghai - Runtime linking replaces compiler-style materialization
  > 1. materialize.cc 's behavior is very wrong. It works in a compiler way: operate on program ir and create sections or other intermediate things (like scheduler_section, schedule-entry). But this is very static, and cause overheads. We don't want to do things like lowering/transform/ir-building. Instead, we just want to link real implementation to APIs and let scheduler/dispatcher and other components run at runtime, not do static compilation. This is fundamentally wrong.
  > 2. the current dispatcher is too ad-hoc.
  > 3. from the ownership's view, we don't want things like program.h records logical facts and detail/materialize.h + src/program/materialize.cc owns copied facts and creates target metadata. This pattern is not as thin as we want. We're not doing compilation!
  - Context: User reviewed the concrete implementation docs and rejected the
    compiler-like implementation shape in PR #3.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/design/concrete_impl/`, `include/megacu/`,
    `src/program/materialize.cc`, PR #3.
  - Agent interpretation: Replace program-IR materialization and static section
    construction with runtime-linked components. CMake should link real
    dispatcher, scheduler, platform, backend, and operator implementations;
    those components should run inside the orchestrate target at runtime.

- 2026-04-25 Asia/Shanghai - Move implementation-ready design back to in-progress
  > Since we've redirecting architectural design, we need to bring docs/design/implementation_ready_device_native_layer back to in_progress and do fixing there first. Also, I found distributed features not covered in the fix plan.
  - Context: User corrected the documentation workflow after rejecting the
    compiler-style implementation shape.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`.
  - Agent interpretation: The implementation-ready design is no longer stable
    implemented behavior and must move back to `docs/in_progress/design/`.
    The fix must happen in design first and must explicitly cover distributed
    runtime support, including CUDA+NVSHMEM under single-card, two-card,
    MPI-launched, and torch-distributed-launched scenarios.

- 2026-04-25 Asia/Shanghai - Recheck runtime-linking design before implementation
  > According to our new requirements, redo, recheck, review, update designs under docs/in_progress/design/implementation_ready_device_native_layer first. The programming surface might also need updating to fit the new requirements.
  - Context: User requested a second design pass after the implementation-ready
    workstream was moved back to in-progress and rewritten around runtime
    linking.
  - Related: `docs/in_progress/design/architecture/`.
  - Agent interpretation: Before changing code again, review and tighten the
    runtime-linking design itself, especially the programming surface, so it
    gives concrete APIs, component boundaries, distributed adapter contracts,
    and verification criteria that match the no-materialization requirement.

- 2026-04-25 Asia/Shanghai - ConfigureTarget owns a general dispatcher
  > You misunderstand me. dispatcher should be part of the configureTarget, but it needs to be more general! And it should expose APIs for programming surface to annotate virtual participents with attributes, and it can do mapping/dispatching according tot he annotations during runtime, with it's algorithm.
  - Context: User corrected the dispatcher/configuration split after the
    runtime-linking redesign risked moving dispatcher ownership away from the
    configured target or making it GEMM+AllReduce-specific.
  - Related: `docs/in_progress/design/architecture/`.
  - Agent interpretation: Dispatcher remains a `ConfigureTarget` component, but
    it must be a general annotation-driven runtime mapper. The programming
    surface should expose APIs for virtual-participant attributes, and the
    linked dispatcher should map ranks, peers, lanes, and work at runtime from
    those annotations plus the current team/problem.

- 2026-04-25 Asia/Shanghai - Rename stale branch and task doc
  > The branch name and docs/in_progress/public_builder_surface.md 's name is stale.
  - Context: User corrected naming after PR #3 moved from public-builder-surface
    work to the runtime-linked Megacu slice.
  - Related: `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: Rename the active task document and branch so they
    describe the runtime-linked Megacu slice instead of the stale public builder
    surface label.

- 2026-04-25 Asia/Shanghai - Rename architecture docs and clarify dispatcher ownership
  > We need to think dispatcher very carefully, since it is responsible for retargeting single-/multiple-card even, required by the gemm-ar example. For gemm-ar-overlap's co-resident, it might alos need to consider? (not sure, maybe scheduler also participates). But before diving into details, we need to make responsibility partition clear at the architecture level. We need to rename docs/in_progress/design/implementation_ready_device_native_layer into **/architecture/ and review it very very carefully.
  - Context: User corrected the active design scope after the dispatcher was
    made a general `ConfigureTarget` component.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #3.
  - Agent interpretation: Rename the active implementation-ready design
    directory to `architecture/` and make the architecture-level responsibility
    partition explicit before further dispatcher details. Dispatcher owns
    spatial/topology retargeting for single-card and multi-card runs; scheduler
    owns temporal progress and participates in overlap co-residency legality.

- 2026-04-25 Asia/Shanghai - Redesign concrete implementation docs after architecture update
  > Redesign/review/update docs/in_progress/design/concrete_impl according to new
  - Context: User requested the implementation-facing docs be brought in line
    after the active design moved to `docs/in_progress/design/architecture/`
    and clarified dispatcher/scheduler/backend responsibility partition.
  - Related: `docs/in_progress/design/concrete_impl/`,
    `docs/in_progress/design/architecture/`, PR #4.
  - Agent interpretation: Rewrite concrete implementation documentation around
    the runtime-linked implementation path. Treat current `program_ir`,
    materialization, static metadata sections, and target-lowering files as
    transitional gaps to replace, not as the desired implementation contract.

- 2026-04-28 Asia/Shanghai - Split PR #4 scope
  > Let's split the PR. PR 4 should focus on the architecture redesign and a very tiny but mighty (can be problem-specific) example without generality concerns. For concrete_impl, they should be moved into todo with strong generality requirements. Then, we go back to concentrate on fixing the architecture design.
  - Context: User corrected PR #4 after the runtime-linked architecture docs
    and concrete implementation docs had grown into one oversized workstream.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/todo/concrete_impl/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #4.
  - Agent interpretation: PR #4 should narrow to architecture repair plus one
    tiny problem-specific proof example. Broad concrete implementation work is
    future TODO work and must carry strong generality requirements before it is
    allowed back into active implementation scope.

- 2026-04-28 Asia/Shanghai - Fix PR #4 programming surface and scope
  > 1. Should our orchestrate ABI to have fixed arguments? This is too restricted for programmers. Also, `cuda_nvshmem_gemm_allreduce_overlap_orchestrate`'s arguments are platform/backend-specific, this is also wierd. You might look at simpler to see how it specifies orch signature.
  > 2. Runtime views are also platform/backend-specific (`cuda::launch_view launch;nvshmem::team_view team;`), which is strange.
  > 3. "The orchestrate call then uses the `ConfigureTarget` dispatcher through a
  > generic API": This is very bad. We don't want to let programmers to write such boilerplates. They might be the execution reality, but shouldn't appear for programming.
  > 4. Participant attribute mechanism should be very flexible and extensible, not fixed things in megacu. The principle is that specific compoenet (dispatcher, scheduler, platform, backend, etc) can provide more attribute option/candidates, and they can be composed during programming and each component might read and utilize the ones they provide. That is, megacu itself only provides interface for components to provide and programmer to write such attributes.
  > 5. Operators also should have configurable arguments in their signature.
  > 6. Another problem is the dependency model between tasks. How megacu handles this?
  > 7. Let's simplify the example in PR4 to be only phased. For overlap case, the kernels with communication should be merged into one fused kernel, rather than several standalone ones to trouble dispatcher and scheduler. However, this conclusion only holds for PR4, and more discussion is needed in future.
  > 8. Both scheduler and dispatcher should be general. But in PR4, they can be minimal and naive.
  > 9. For the PR4, make components in docs/in_progress/design/architecture/03-runtime-components.md tiny but mighty. No need to give so many participant attributes, since we need to change the mechanism.
  > 10. No need to cover MPI and torch-distributed in PR4.
  > 11. For PR4, we need 1-host-1-device and 1-host-2-device for implementation and example.
  - Context: User reviewed the split PR #4 architecture docs and identified
    remaining over-specific ABI, runtime-view, attribute, dispatcher/scheduler,
    operator, dependency, and example-scope problems.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`,
    `docs/notes/orchestration_surface_sources.md`, PR #4.
  - Agent interpretation: The programming surface must be target-argument
    configurable and avoid platform/backend-specific public arguments. Runtime
    component calls are execution reality, not author boilerplate. Attributes
    are component-provided and composable, not fixed Megacu enums. Operators
    need configurable signatures. Task dependencies must be modeled explicitly.
    PR #4 should implement only the phased tiny example with minimal general
    dispatcher/scheduler components and both 1-host/1-device and
    1-host/2-device coverage; MPI, torch-distributed, and overlap are future
    work.

- 2026-04-29 Asia/Shanghai - Kernel-like orchestrate arguments and explicit dependency attributes
  > Look at the new design. I found the problems:
  > 1. What should be the arguments? We need to be very clear about how megacu things should be used. For a specific platform + backend, using CUDA + NVSHMEM as the example, the megacu orchestrate is called from the host code. Generally, host code includes Asynchronous Execution things like CUDA streams to dispatch and synchronize tasks, where the orchestrate will become ONE mega-kernel. So the arguments should be like arguments that a CUDA kernel can take, and we don't need to over-wrap them. We should also make it very clear about the abstraction of driver (especially for distributed execution).
  > 2. Of course scheduler or dispatcher require dependencies between tasks. However, we don't use the way like simpler's tensormap_and_ringbuffer runtime, where dependencies are determined by looking up operations on tensors. Instead, megacu should be very thin to only treat dependencies as one kind of attributes in the explicit way.
  - Context: User reviewed the active PR #4 runtime-linked architecture draft
    after it introduced `orchestrate_args`, `target_env`, and dependency
    derivation from submitted tensor accesses.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`,
    `docs/todo/concrete_impl/README.md`, PR #4.
  - Agent interpretation: Public orchestrate entries should look like host
    functions that enqueue one target megakernel using a driver plus
    CUDA-kernel-like payload arguments, not an over-wrapped generic argument
    frame. The driver abstraction owns asynchronous execution and distributed
    platform/backend resources. Task inputs, outputs, and inouts describe
    operator arguments, but scheduling dependencies must be explicit typed
    attributes rather than inferred by looking up tensor operations.

- 2026-04-29 Asia/Shanghai - Raw task arguments without input/output/inout abstraction
  > I don't think tasks need to have explicit input, output, inout. Such pointers can be passed raw, without a new arg abstraction.
  - Context: User corrected the active PR #4 programming-surface draft after it
    replaced inferred dependencies with explicit dependency attributes but still
    modeled task arguments through `input`, `output`, `inout`, and `op_args`.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`,
    `docs/todo/concrete_impl/README.md`, PR #4.
  - Agent interpretation: Task submission should not introduce a Megacu
    input/output/inout argument abstraction. The orchestrator should pass raw
    pointers, scalars, or small descriptors directly to linked operators, while
    dependencies and other scheduling/dispatch/backend facts remain explicit
    typed attributes.

- 2026-04-29 Asia/Shanghai - Remove payload argument as a design concept
  > Do we still need the payload argument?

  > Yes go.
  - Context: User questioned whether the active PR #4 design still needed the
    `payload argument` term after task arguments were reduced to raw pointers,
    scalars, and descriptors, then approved removing that concept.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`,
    `docs/todo/concrete_impl/README.md`, PR #4.
  - Agent interpretation: The public model should be `orchestrate(driver,
    target_arg0, target_arg1, ...)`. `driver` is the special execution object;
    all following values are normal target arguments shaped like CUDA kernel
    arguments. Public docs should use terms such as target arguments, target
    signature, or target argument schema instead of payload arguments.

- 2026-04-29 Asia/Shanghai - Review fixes for driver, notes, megakernel, submit, and dependencies
  > 1. Explicitly mark outdate notes that contradict with current design as rejected.
  > 2. Remove them clearly. Also, avoid any unnecessary checking.
  > 3. Discuss carefully about what should be included in the driver and why.
  > 4. The one mega-kernel includes multiple kernels inside. The kernels are dispatched and scheduled by the megakernel.
  > 5. Pick the thin option.
  > 6. Let user take responsibility about avoid missing things. Megacu does not do any fallback checking.
  > 7. Update.
  - Context: User reviewed the architecture-risk report and selected concrete
    fixes for the remaining design risks.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`,
    `docs/notes/orchestration_surface_sources.md`,
    `docs/todo/concrete_impl/README.md`, PR #4.
  - Agent interpretation: Supersede contradictory source-note conclusions,
    narrow `driver` to execution/distributed resources, define the Megacu
    megakernel as one target operation that may dispatch and schedule multiple
    native kernels internally, parse `submit` by linked operator signature
    arity, and make missing explicit dependencies the user's responsibility
    rather than a Megacu fallback-checking obligation.

- 2026-04-29 Asia/Shanghai - Standalone intuitive flow explanation
  > I need a very intuitive explanation (put in a standalone file) about the program, compilation, execution all flow, describing how megacu is used to construct the megakernel from invocations of small kernel operators, and make it runnable on a distributed environment (CUDA + NVSHMEM).
  - Context: User requested a standalone explanation after the runtime-linked
    architecture was corrected around driver, direct target arguments, raw
    operator arguments, explicit dependency attributes, and one Megacu
    megakernel that internally dispatches native kernels.
  - Related: `docs/in_progress/design/architecture/`, PR #4.
  - Agent interpretation: Add an intuitive narrative document that explains
    authoring/programming, CMake compile/link, and runtime execution for the
    CUDA+NVSHMEM distributed path without introducing new abstractions.

- 2026-04-30 Asia/Shanghai - Implement PR reset architecture
  > Let's move on the implementation of the PR's reset architecture. The existing codebase should be fully reset and moved towards the new architecture.
  - Context: User approved the corrected runtime-linked architecture direction
    and asked to start replacing the existing compiler-like implementation.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #4.
  - Agent interpretation: Replace the current program-IR/materialization/static
    metadata implementation path with a runtime-linked path around direct target
    arguments, a narrow driver, raw operator submissions, explicit dependency
    attributes, and linked runtime components.

- 2026-04-30 Asia/Shanghai - Merge gate requires runnable GEMM-AR examples and strict review
  > For the PR's merging, we need: 1. an expected example to run successfully on the GEMM-AR for both 1-host-1-gpu and 1-host-2-gpu, with the expected architecture; 2. perfect review on the codebase and example code to meet the design.
  - Context: User clarified the merge-readiness bar after the reset
    architecture implementation pass.
  - Related: `docs/in_progress/design/architecture/`,
    `docs/in_progress/runtime_linked_megacu_slice.md`, PR #4.
  - Agent interpretation: PR #4 should not be merged on build-only evidence.
    It needs runnable GEMM+AllReduce evidence for both single-GPU and two-GPU
    local host configurations, and a strict review of code and examples against
    the runtime-linked design contract.

- 2026-04-30 Asia/Shanghai - Remove overlap GEMM-AR example from PR4
  > Remove any overlap gemm-ar example since we don't need it in PR4.
  - Context: User tightened PR #4 scope after merge-readiness review found
    leftover overlap example surfaces.
  - Related: `examples/cuda_nvshmem/gemm_allreduce/`,
    `docs/in_progress/design/architecture/`, PR #4.
  - Agent interpretation: Remove overlap GEMM+AllReduce example code and docs
    from the active PR #4 tree rather than leaving placeholder or baseline
    examples that imply overlap is part of the merge gate.

- 2026-04-30 Asia/Shanghai - Current implementation is too fake
  > The implementation of megacu looks completely fake yet. We need real, working implementation.
  - Context: User reviewed the reset implementation and found that the API and
    tests were still too close to link anchors and golden-wrapper delegation.
  - Related: `include/megacu/runtime.h`,
    `examples/cuda_nvshmem/gemm_allreduce/`, PR #4.
  - Agent interpretation: Replace no-op runtime behavior with a real minimal
    task execution path, and make the GEMM+AllReduce Megacu example execute
    through submitted linked operators instead of bypassing Megacu through
    golden wrappers.

- 2026-04-30 Asia/Shanghai - Megacu phased README must show run environment
  > examples/cuda_nvshmem/gemm_allreduce/phased/megacu/README.md doesn't make it clear how to run the compiled binaries with the environment setup.
  - Context: User reviewed the phased Megacu example documentation after the
    real CUDA+NVSHMEM implementation was added.
  - Related:
    `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/README.md`,
    `tools/cuda_nvshmem/run_two_card_docker.sh`, PR #4.
  - Agent interpretation: The example README must show direct executable paths
    and environment setup for CUDA single-process runs and two-rank NVSHMEM
    runs, not only CTest commands.

- 2026-04-30 Asia/Shanghai - PR4 keeps platform/backend validation thin
  > in this pr, we dont want the heavy validation logic at all. other fixes are as you suggest.
  - Context: User reviewed the code/design review findings about scheduler
    generality, dispatcher/component reality, platform/backend shortness, and
    validation scope.
  - Related: `include/megacu/runtime.h`, `src/dispatcher/`,
    `src/scheduler/`, `src/platform/cuda/`, `src/backends/nvshmem/`,
    `docs/in_progress/design/architecture/`, PR #4.
  - Agent interpretation: Do not add a heavy platform/backend validation layer
    in PR #4. Keep platform/backend as thin execution-fact helpers, while
    still making dispatcher, scheduler, and target runtime real general
    components instead of GEMM+AllReduce-specific or link-marker-only code.

- 2026-05-02 Asia/Shanghai - Add native sync-only tasks from Event Tensor gap
  > one missing feature compared to event tensor is that it can flexibily define a task to be non-logic but only for sync. Reread the event tensor paper/notes, to make sure we have it in megacu in a seamless/native way.
  >
  > good
  >
  > go agead
  - Context: User reviewed the Megacu/Event Tensor alignment and approved the
    design that represents Event Tensor's middle readiness node as a native
    sync-only task.
  - Related: `include/megacu/runtime.h`,
    `src/scheduler/explicit_phase.cc`,
    `docs/in_progress/design/architecture/03-runtime-components.md`,
    `docs/notes/megakernel_cuda_layer_sources.md`, PR #4.
  - Agent interpretation: Megacu should support first-class sync-only tasks
    that return `task_ref`, carry explicit dependency/sync attributes, and
    invoke no operator body. Users still do not program scheduler calls, waits,
    notifies, or backend-specific synchronization intrinsics; those remain
    Megacu internal platform/backend mechanisms.

- 2026-05-03 Asia/Shanghai - Event Tensor must be orchestration-owned attrs
  > I think the co-called "runtime-owned event stage kernels" are not good design. Instead, similar to Event Tensor, we should define "event tensor" and the related APIs as orch owned ones. Can this be added as handlers on tasks, using the existing attrs mechanism? That is, one platform/backend can provide attributes about the event tensor operations for orch to use, and orch should use the attrs when submitting tasks, which get lowerered to the sync operations.
  >
  > This is good.
  - Context: User corrected the Event Tensor design before implementation of
    PR #4 sync lowering.
  - Related: `include/megacu/runtime.h`,
    `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h`,
    `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu`,
    `docs/in_progress/design/architecture/03-runtime-components.md`, PR #4.
  - Agent interpretation: Reject runtime-owned event stage kernels. The
    orchestrator should declare event tensors and attach event operations as
    task attrs, while platform/backend components lower those attrs to concrete
    CUDA+NVSHMEM waits, signals, fences, and remote operations inside the
    mega-kernel. Operator bodies should remain free of readiness logic.

- 2026-05-03 Asia/Shanghai - Manual mega-kernel belongs to baseline, not Megacu
  > we should use the manual-mega-kernel as another baseline, while the megacu one should be composed from operators as tasks into one megakernel.
  - Context: User reviewed the PR #4 CUDA+NVSHMEM implementation after the
    Event Tensor attr direction was added.
  - Related:
    `examples/cuda_nvshmem/gemm_allreduce/phased/baseline/`,
    `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/`, PR #4.
  - Agent interpretation: The handwritten fused mega-kernel is valid as a
    baseline artifact only. The Megacu path should show composition from
    submitted operator tasks plus event tensor handlers into one mega-kernel.

- 2026-05-03 Asia/Shanghai - Composer must not be platform ASAP code
  > yes, the composer should be part of the target (carefully decode where it should locate), not programmed algo in example.
  >
  > include/megacu/platform/cuda/asap_tile_composer.cuh looks bad, since asap belongs to scheduler, not what platform should own. I don't understand why we need such a very specific kernel entry, not a general one.
  >
  > good.
  - Context: User clarified the ownership boundary after the manual
    mega-kernel was split into the baseline, then corrected the first
    target-owned composer location proposal.
  - Related: `include/megacu/platform/cuda/megakernel.cuh`,
    `include/megacu/backends/nvshmem/cuda_event_tensor.cuh`,
    `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/`, PR #4.
  - Agent interpretation: Do not put ASAP policy in platform code. CUDA
    platform should own only a generic mega-kernel launch shell. NVSHMEM
    backend should own the CUDA-backed event tensor operations. The common
    device entry should own the running loop that uses those pieces, while the
    platform entry itself must stay general.

- 2026-05-04 Asia/Shanghai - Megakernel running logic must be common
  > The megakernel doesn't have a general running logic now. It lives in gemm_allreduce_program (examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu), which is still problem-specific. Let's careful discuss about the pattern here. What should be common? I think megakernel things should be common. It is the scheduler that determines what to run next. And the kernel entry just run the operators in the order that scheduler dynamically gives and on the device/grid/threadblock that dispatcher determiens.
  >
  > This is better. But the make_device_program looks tedious. Instead, dispatcher, scheduler, should be linked and the common API/signature should be directly used.
  >
  > Good.
  - Context: User reviewed the refactor that moved the CUDA mega-kernel shell
    and NVSHMEM event tensor into platform/backend headers, then identified
    the remaining problem-specific running loop.
  - Related: `include/megacu/runtime/device_entry.cuh`,
    `include/megacu/scheduler/explicit_asap_device.cuh`,
    `include/megacu/dispatcher/tile_grid_device.cuh`,
    `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu`,
    PR #4.
  - Agent interpretation: The common mega-kernel device entry should own the
    running loop. It should call linked scheduler, dispatcher, backend, and
    operator-table APIs directly. Avoid a user-facing `make_device_program`
    builder. The example should instantiate linked components and provide an
    operator table, not own the scheduler/dispatcher execution algorithm.
