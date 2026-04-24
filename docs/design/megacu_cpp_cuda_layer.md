# Design: Megacu Device-Native Layer

This is the stable entry point for the picked Megacu device-native layer
direction.

Implementation-ready contracts for the first implementation live under
`docs/design/implementation_ready_device_native_layer/`.

## Accepted Direction

Megacu uses this public lifecycle:

1. authored orchestrate program
2. CMake target
3. run

The key implementation rule is:

> CMake/build belongs to offline native build flow, not to the runtime C++ API.

That means:

- runtime C++ does not expose build steps;
- reusable dispatcher, scheduler, kernel lowering, platform, and backend
  implementations are organized as reusable build targets;
- the authored orchestrate program is compiled/linked against those artifacts;
- runtime C++ runs the compiled orchestration directly.

## Language Split

The current design assumes three implementation surfaces:

- **Authoring C++**: author the orchestrate program, register kernels, and
  express orchestration through the thin API.
- **CMake/native build system**: build reusable component targets, then
  compile/link the orchestrate target against them.
- **Runtime C++/CUDA**: call the compiled orchestration.

This preserves the thin native hot path while keeping materialization and
packaging outside the runtime process.

## Source Context

- MPK and Event Tensor source reading:
  `docs/notes/megakernel_cuda_layer_sources.md`
- Triton-distributed, MegaKittens, MSCCL++, and UniEP source reading:
  `docs/notes/distributed_backend_sources.md`
- FlashInfer framework-integration source reading:
  `docs/notes/framework_integration_sources.md`
- PTO Runtime / simpler orchestration-surface reading:
  `docs/notes/orchestration_surface_sources.md`

## Accepted Boundaries

The accepted design keeps strategy choices out of the runtime API:

- no public task-trait authoring model;
- no runtime-selected profile/configuration object;
- no public scheduler families in the thin core;
- no public fragment-op taxonomy;
- no runtime C++ build entry point;
- no first-party Python dependency in Megacu itself;
- no string-path loading and no generic `runtime_env` in the primary API.
