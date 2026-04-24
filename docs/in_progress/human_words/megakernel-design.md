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
    `docs/in_progress/design/implementation_ready_device_native_layer/`,
    `docs/in_progress/implementation_ready_device_native_design.md`
  - Agent interpretation: Keep `docs/design/` as a concise picked-direction
    snapshot, remove unfinished detailed chapters from stable design docs, do
    concrete refinement only under `docs/in_progress/design/`, and merge
    accepted documents back into `docs/design/` at PR closeout.

- 2026-04-24 Asia/Shanghai - Component-owned implementation contracts
  > We shouldn't put every thing in 11-implemented-contracts.md. Instead, we need to make very component of the design concrete, including the core abstraction APIs (event, task, ...) and all needs to be implemented.
  - Context: User reviewed the first PR slice and rejected concentrating
    implementation detail in one catch-all contract chapter.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Move concrete APIs, planned paths, records, and
    evidence into the owning component chapters. The active design should make
    each implementable component concrete instead of centralizing all detail in
    a final implementation-contract appendix.

- 2026-04-24 Asia/Shanghai - Explain orchestrator, operators, execution, and terms
  > I found things /APIs are not concrete enough. Fundamentally, we need to make it clear what we program in orchestrator and kernels (operators), and how they will be runned. For example, when we write event (with a name) or call primitives in a kernel for communication with another kernel (identified by like a virtual id?), how they will be runned with backend things provided (like providing how virtual id is resolved and event names are resolved). Also, all implementation APIs are quite opaque still. Another problem is that we don't give enough explanation for concepts/terms yet. For example, I just cannot understand what the workspace or domain mean.
  - Context: User reviewed the component-owned contracts and found that the
    design still lacked a concrete execution model and term definitions.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Define the vocabulary before APIs, distinguish
    diagnostic labels from typed tags, specify what users program in the
    orchestrator versus kernels, and document how lowering resolves domains,
    virtual participants, event tags, workspace views, and backend handles into
    runnable CUDA/NVSHMEM behavior.

- 2026-04-24 Asia/Shanghai - Clarify tiles, tile, workspace, events, bind, and extent
  > You should make it more cler about that is tiles, tile, workspace, events, and what exec.bind , megacu::extent are.
  - Context: User reviewed the descriptor/executor model and called out
    specific opaque terms in the first API example.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`
  - Agent interpretation: Explain `tiles` as runtime extent count, `tile` as
    logical domain, `workspace` as caller-owned payload/scratch storage,
    `events` as caller-owned synchronization storage, `exec.bind` as typed slot
    binding, and `megacu::extent` as typed runtime extent binding.

- 2026-04-24 Asia/Shanghai - Require necessity analysis
  > are these concepts and binding behavior necessary? Give rationale or necessity analysis.
  - Context: User asked whether the newly clarified concepts and runtime
    binding behavior are justified or removable.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Add explicit necessity analysis for each public
    concept and compare typed runtime binding against generic env bags,
    positional arguments, rebuild-per-shape, raw kernel plumbing, and
    Megacu-owned allocation.

- 2026-04-24 Asia/Shanghai - Dispatcher owns virtual participant mapping
  > This is weird. Why config in CMake? Why not do in dispatcher?
  - Context: User pointed at `MAP producer_lane TO RANK 0` /
    `MAP consumer_lane TO RANK 1` in the active design and questioned why
    participant placement was represented as CMake configuration.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`
  - Agent interpretation: Virtual participants are logical role declarations.
    CMake may select the dispatcher component, but dispatcher policy owns
    participant-to-backend placement and emits participant mapping metadata.

- 2026-04-24 Asia/Shanghai - Parameterized compiled orchestrate function
  > I found the split between orchestrate-target structure and run-time dynamics not very necessary. Why not made orchestrate just to be a parameterized one? Why we still need exec.bind and exec.run upon compiled orch?
  - Context: User questioned the public `executor` / `bind` / `run` layer after
    the target has already been compiled.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Keep descriptor slots as lowering internals, but make
    the public compiled target a normal parameterized orchestrate function.
    Runtime parameter binding and fast-path `run` are generated/linked target
    internals, not public post-compile APIs.

- 2026-04-24 Asia/Shanghai - Lowering links existing implementations
  > We need to make sure things like kernel-lowering and target-lowering doesn't emit/generate new code. Instead, it mainly link the required low-level implementation to the high-level APIs as the backend provides.
  > We need to specify it very clear about what megacu's compilation do (compile, link) and not do (translation into new CUDA or other language code).
  - Context: User clarified that target/kernel lowering should not be a source
    generation or translation pipeline.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/02-principles-and-naming.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/05-language-responsibilities.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Megacu compilation means native compile/link plus
    compact metadata materialization. Kernel lowering and target lowering must
    select and link reusable dispatcher, scheduler, lowering, platform,
    backend, and kernel implementations; they must not emit new C++/CUDA or
    translate the orchestrate program into another language.

- 2026-04-24 Asia/Shanghai - Strengthen examples with GEMM+AllReduce and MPK
  > The example is too weak. We need a gemm-allreduce fusion-kernel as the example. You can get information and reference code from related repositories. Also, we need another bigger example from mpk (where cuda is provided).
  - Context: User reviewed the event-copy example and found it insufficient for
    implementation-ready design.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`,
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
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/00-overview.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/03-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/04-cmake-build-and-runtime.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/06-compiled-orchestrate-program.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/07-dispatcher-scheduler-kernel.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`
  - Agent interpretation: Implementation-ready design must include concrete
    records, owner paths, slot binding, materialization flow, function-body
    shape, and metadata checks, not only concept explanations and examples.

- 2026-04-24 Asia/Shanghai - Restrict MPK source evidence
  > For example creation, we need to make it clear that not everything inside mpk repo relates to mpk, since the original mirage things are outdated but still kept in the repo. Only things in research/repos/mirage-mpk/src/kernel and research/repos/mirage-mpk/python/mirage/mpk should be thought about as mpk-related.
  - Context: User corrected the source boundary for MPK-based examples and
    design evidence.
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
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
  - Related: `docs/in_progress/design/implementation_ready_device_native_layer/11-distributed-launch-and-framework-integration.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/08-examples.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/09-first-validation-slice.md`,
    `docs/in_progress/design/implementation_ready_device_native_layer/10-verification.md`,
    `docs/notes/distributed_launch_sources.md`
  - Agent interpretation: The design must specify how Torch Distributed and MPI
    process models construct CUDA launch views, NVSHMEM team views, symmetric
    allocations, and validation evidence before the compiled orchestrate target
    runs.
