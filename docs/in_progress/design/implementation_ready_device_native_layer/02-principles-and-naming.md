# Principles And Naming

## Core Naming Model

Megacu should use one clear public lifecycle:

- authored orchestrate program
- compiled CMake target
- run

This replaces earlier `profile`, `prepare`, `bind`, explicit
`compile/assemble`, and generated runtime-entrypoint vocabulary.

Reasons:

- `profile` was too vague about where strategy choice lives;
- `prepare` blurred build-time and runtime work;
- `bind` understated what runtime setup really does;
- `compile/assemble` is useful as an internal build-graph structure, but too
  heavy as the primary user-facing lifecycle;
- generated runtime entrypoints were still too indirect compared with simply
  compiling and calling the authored orchestration.

Megacu compilation must mean native compile and link, not translation into a
new generated CUDA/C++/Triton/etc. program. The build may materialize compact
target metadata, but the executable code comes from user-authored C++/CUDA
kernels and reusable Megacu/backend/platform implementations.

## Thin Core Principle

The thin core owns only what must remain visible in the authored orchestration:

- named operations
- resources
- explicit events
- logical work domains
- orchestration order and control flow

It does not own public families of:

- scheduler APIs
- kernel-shape APIs
- backend capability taxonomies
- raw descriptor authoring APIs
- raw worker/rank/coordinate id types as the normal authoring surface

## CMake-Managed Build Graph Principle

Dispatcher, scheduler, kernel lowering, platform, and backend are not runtime
choices in the C++ API.

The design should let CMake own the offline build graph. That build graph may
internally separate:

1. reusable component targets compiled once
2. orchestrate-program targets that reuse those artifacts

That means:

- the runtime does not choose strategy;
- the compiled orchestrate program already has its strategy fixed by the build
  graph;
- runtime work is limited to concrete setup and repeated execution inside the
  validated envelope;
- expensive reusable compilation can be cached independently from program
  compilation/linking.

## Single Named-Op Principle

Megacu should not expose multiple public operation kinds such as
`external_kernel`, `device_fragment`, and `backend_primitive`.

The public model should have one operation concept: a named op.

Fine-grained compute/communication overlap is still supported because:

- named kernels may call backend primitives inside the kernel body;
- build-time kernel-lowering engines may stitch or compose named kernels into a
  persistent execution form when the selected target configuration supports it.

The distinction between "plain kernel" and "stitched kernel fragment" is
lowering metadata, not public API taxonomy.

## C++-Only Project Principle

Megacu itself should be C++/CUDA only.

That means:

- no first-party Python package is required for authoring;
- no Python build driver is required to produce targets;
- no Python dependency is part of the core Megacu design.

This does not forbid external framework wrappers from existing outside the core
project boundary.

## Direct Compiled Runtime Principle

The normal runtime path should not require:

- `load_execution_module("path/to/module.so")`
- generic `runtime_env` bags
- string-based resource lookup
- an extra generated wrapper entrypoint when the authored orchestrate program
  itself can be compiled and called

Instead, normal code should compile the authored orchestration with CMake and
call it directly.

This makes the required resources explicit, avoids string/path plumbing in
ordinary user code, and better fits the zero-overhead claim.

## CMake Owns The Build Step

The user concern is correct: build steps should not appear in the runtime C++
API.

The design should therefore assume:

- reusable component compilation and orchestrate-program compilation/linking are
  driven by CMake/native build rules;
- C++/CUDA is used for kernel authoring, metadata materialization, and runtime
  execution, but not for invoking build steps through the runtime API;
- deployment-side C++ code links the compiled orchestrate program and calls it.

This does not forbid C++ from participating in the build system. It only means
that runtime C++ APIs should not expose "build this program now" behavior.

## Related Works As Pressure Tests

Megacu does not aim to copy every mechanism used by MPK, Event Tensor,
Triton-distributed, MegaKittens, FlashInfer, or PTO Runtime.

Those systems are pressure tests for whether the design is:

- expressive enough
- thin enough
- inspectable enough
- practical to integrate

Coverage should come from a coherent Megacu model, not from public compatibility
with each system's internal mechanism names.
