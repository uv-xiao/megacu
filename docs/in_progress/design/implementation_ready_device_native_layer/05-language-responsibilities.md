# Language Responsibilities

This chapter answers the implementation question directly: what should be done
in authoring C++, what should be done in CMake/native build rules, and what
should be done in runtime C++/CUDA?

## Authoring C++ Layer

Authoring C++ is the right place for:

- authoring the orchestrate program
- defining explicit runtime argument types
- registering named kernels and native adapters
- expressing the orchestration through a small builder surface

This is the natural place for the current design because:

- the project should stay C++ only
- CMake should own build graph work outside the runtime API
- the program model should stay close to ordinary native code rather than a
  template-heavy metaprogramming layer

This authoring code is the thing that gets compiled and run.

## CMake And Native Build Layer

CMake/native build rules are a separate responsibility from both authoring C++
and runtime C++.

They are the right place for:

- compiling reusable dispatcher/scheduler/kernel-lowering targets
- compiling reusable kernels and helper code
- compiling/linking the authored orchestrate program target
- executing any lowering/codegen needed by the chosen backend
- linking platform/backend adapters
- producing optional inspection metadata

In other words, Megacu should separate:

1. reusable compilation
2. later orchestrate-program compilation/linking

Both are offline native build-graph steps, and both stay out of the runtime
API.

## C++ And CUDA Layer

C++/CUDA is the right place for:

- named kernel implementations
- backend primitive headers and device calls
- platform adapters
- backend adapters
- generated lowering code emitted during the build
- the authored orchestrate program
- the internal `run(...)` path

This keeps the hot path close to native code and lets the orchestrate program
be a regular compiled target.

## Runtime C++ API

The runtime C++ API should be small:

- call the compiled orchestrate program

It should not:

- expose CMake/build steps
- expose strategy selection
- expose packaging concerns that belong to the offline build path

That is the core boundary for implementation.

## What About Pure C++ Users?

Pure C++ users are the primary path, but still through an offline CMake/native
build path, not through a runtime build API.

Possible working paths:

- use CMake helper functions such as `megacu_add_components(...)` and
  `megacu_add_orchestrate_target(...)`
- use Bazel rules that mirror the same separation
- check in prebuilt runtime artifacts when deployment needs that
- link the resulting target and call the orchestrate function from C++

So the absence of runtime build APIs does not mean C++ users are excluded. It
only means strategy selection stays outside the runtime process.

## First Implementation Recommendation

For the first implementation, the cleanest split is:

- **CMake/native build rules**
  - reusable component-target hooks
  - orchestrate-target hooks
  - execution of any lowering/codegen needed by the backend
  - CUDA/C++ compilation and linking

- **C++/CUDA**
  - orchestrate-program authoring
  - kernels
  - generated lowering code
  - `run(...)`

This aligns the implementation path with the design goal of compile-time
strategy selection and runtime minimal overhead.
