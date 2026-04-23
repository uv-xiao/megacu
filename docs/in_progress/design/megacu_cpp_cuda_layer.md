# Design: Megacu Device-Native Layer

This is the stable entry point for the active Megacu design.

The detailed design lives under `megacu_device_native_layer/` as one flat,
ordered chapter set. Start at
`docs/in_progress/design/megacu_device_native_layer/00-overview.md` and read
forward numerically.

## Active Direction

Megacu is being redesigned around this public lifecycle:

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

## Ordered Design Files

- `megacu_device_native_layer/00-overview.md`
- `megacu_device_native_layer/01-positioning.md`
- `megacu_device_native_layer/02-principles-and-naming.md`
- `megacu_device_native_layer/03-program.md`
- `megacu_device_native_layer/04-cmake-build-and-runtime.md`
- `megacu_device_native_layer/05-language-responsibilities.md`
- `megacu_device_native_layer/06-compiled-orchestrate-program.md`
- `megacu_device_native_layer/07-dispatcher-scheduler-kernel.md`
- `megacu_device_native_layer/08-examples.md`
- `megacu_device_native_layer/09-first-validation-slice.md`
- `megacu_device_native_layer/10-verification.md`
- `megacu_device_native_layer/90-redesign-2026-04-24-cmake-build-and-direct-runtime.md`
- `megacu_device_native_layer/91-redesign-protocol.md`

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

## Current Review Focus

The redesign is removing public surface and pushing strategy choices out of the
runtime API:

- no public task-trait authoring model;
- no runtime-selected profile/configuration object;
- no public scheduler families in the thin core;
- no public fragment-op taxonomy;
- no runtime C++ build entry point;
- no first-party Python dependency in Megacu itself;
- no string-path loading and no generic `runtime_env` in the primary API.
