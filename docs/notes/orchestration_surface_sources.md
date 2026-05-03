# Orchestration Surface Source Notes

- Date: 2026-04-23 Asia/Shanghai
- Purpose: inform Megacu's redesign toward a thinner author-facing
  orchestration API and a clearer separation between core semantics and
  configuration-selected runtime behavior.
- Related design: `docs/design/megacu_cpp_cuda_layer.md`

## Sources Read

### Simpler / PTO Runtime

- Repository: `research/repos/simpler`
- Upstream: https://github.com/hw-native-sys/simpler
- Local revision: shallow clone of `main` on 2026-04-23
- Files read:
  - `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
  - `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
  - `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
  - `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_submit_types.h`

## What Simpler Contributes

The `paged_attention_orch.cpp` example is useful because the orchestration code
is ordinary C++ control flow instead of a template-heavy task-definition layer.
The author-facing surface is narrow:

- read input tensor metadata and scalars from a small runtime argument object;
- create external tensor views and lightweight temporary tensors;
- build an `Arg` object by calling `add_input`, `add_output`, and `add_inout`;
- submit concrete kernels by id through small runtime helpers;
- rely on the selected runtime to allocate buffers, build dependencies, and
  schedule work.

The orchestration surface stays simple because the complex parts live in the
runtime/configuration side:

- kernel-id registry and dispatch tables;
- scheduler implementation;
- dependency and tensor-map machinery;
- resource-shape mapping;
- runtime-owned buffer allocation;
- device/host execution plumbing.

Megacu should not copy PTO's internals or dependency builder, but this split is
good evidence for the desired API shape: the user-facing orchestration layer can
stay concise if the runtime/configuration layer owns lowering details.

## Megacu Design Lessons

- The core authoring surface should look like ordinary orchestration code, not a
  meta-language built around `task_traits`.
- Users should submit registered operations through small APIs instead of
  manually constructing low-level descriptors.
- Raw descriptors are execution artifacts built during prepare/lower, not the
  primary authoring surface.
- Scheduler choice, kernel shape, platform binding, and backend binding should
  be selected by configuration profiles or linked implementations, not by
  exposing multiple unrelated core APIs.
- Kernel authors should still be able to write raw CUDA/native kernels; the thin
  orchestration layer should reference those kernels through configuration
  registries and compact operation handles.

## 2026-04-29 Superseded PR #4 ABI Conclusions

The 2026-04-28 ABI conclusions from the Simpler/PTO reading are rejected where
they contradict the active runtime-linked Megacu design. They are preserved here
only as source-reading history, not as current design evidence.

Rejected conclusions:

- Megacu should use a compact frame-like object for target arguments.
- Megacu task submission should use `input`/`output`/`inout` argument builders
  or typed wrappers as the public task argument abstraction.
- Megacu should infer dependencies from input/output/inout access.

Current active conclusion:

- Megacu orchestrate is a host-called function:
  `orchestrate(driver, target_arg0, target_arg1, ...)`.
- The driver owns platform/backend execution and distributed resources, not
  target arguments or linked component state.
- Target and operator arguments are passed as raw pointers, scalars, or small
  descriptors shaped like CUDA kernel arguments.
- Dependencies are explicit attributes such as `scheduler::depends_on(...)`.
  Megacu does not infer dependency edges from tensor access or pointer aliasing.
