# Redesign Protocol

Each redesign pass should start from the current public lifecycle:

`authored orchestrate program -> CMake target -> run`

The first question is always:

> Does this concept belong in the authored orchestrate program, in the
> CMake/build graph, in target internals, or only in runtime execution?

## Keep The Core Small

A concept belongs in the thin core only if removing it would make the authored
program unable to express:

- which named op runs
- which resources it uses
- which explicit event it waits on or signals
- what logical work domain it belongs to
- what orchestration order or control flow exists between ops

If a concept is easier to explain as dispatcher metadata, scheduler state,
kernel-lowering metadata, CMake/build configuration, or target internals, it
should move out of the thin core.

## Keep CMake Out Of Runtime C++

Each iteration must also ask:

- can this step be moved into the CMake/build graph?
- should it live in reusable component targets or orchestrate-target
  compilation/linking?
- does the runtime C++ API really need to see this choice?

The default answer should be:

- reusable engines belong in component targets;
- concrete orchestration lowering belongs in orchestrate-target compilation;
- runtime C++ only calls the compiled orchestrate program.

## Redesign Steps

1. **Redundancy scan**
   Find concepts that are duplicated, too low level for users, or better owned
   by the build graph.

2. **Program/build/runtime split**
   Decide whether each concept belongs to the authored program, build-graph
   generation, target internals, or runtime execution.

3. **Language split**
   Recheck whether the work should be done in authoring C++, in CMake/native
   build rules, in C++/CUDA build targets, or in runtime C++ APIs.

4. **Runtime minimization**
   Make the runtime path be the compiled orchestration itself, and keep the
   internal `run(...)` path as the repeated fast path.

5. **Proof update**
   Update examples, validation slice, and verification so they match the new
   lifecycle and build/runtime split.

## Stop Rule

Stop when the design can be explained simply:

- authoring C++ defines the orchestrate program
- CMake/native rules build reusable components and compile/link the
  orchestrate target
- C++/CUDA provides kernels and native backend/platform code
- runtime calls the compiled orchestration

If the design still requires the runtime C++ API to explain CMake/build
strategy selection, the redesign is not done.
