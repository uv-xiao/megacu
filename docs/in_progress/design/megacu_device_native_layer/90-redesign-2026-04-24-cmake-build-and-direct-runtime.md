# Redesign Note: CMake Build And Direct Runtime

## Why This Redesign Happened

The earlier redesign solved some boundary problems, but the primary runtime
story still looked too indirect:

- explicit generated runtime entrypoints as the main example
- residual plugin-style vocabulary
- a gap between the authored orchestration and the thing users actually call

User review made the target clearer: CMake should own the build graph, reusable
artifacts should still be reused, and the authored orchestrate program itself
should be the thing that gets compiled and called.

## Main Decisions

- The core authoring model is now an authored orchestrate program plus small
  argument builders.
- Scheduler choice, dispatcher choice, kernel lowering, platform binding,
  backend binding, and id mapping are CMake/build-graph concerns.
- Descriptors remain necessary, but they are target internals rather than the
  primary user-authoring surface.
- The public lifecycle is now authored orchestrate program -> CMake target ->
  run.
- The build graph can still separate reusable component targets from
  orchestrate-target compilation/linking, but that is not the primary runtime
  vocabulary.
- The normal runtime path calls the compiled orchestrate program directly, not a
  generated wrapper and not a generic runtime env.
- The design chapters are flattened and numbered so review order is explicit.

## Evidence Used

- FlashInfer remains the main framework-integration evidence.
- PTO Runtime / `simpler` remains useful evidence for the desired author-facing
  orchestration shape: ordinary control flow plus small argument builders, with
  complexity owned by the build/runtime side rather than the user-facing API.

## Directory Cleanup

The earlier fine-grained iteration notes were removed because they preserved a
superseded abstraction path. This redesign note now records the active turn,
while the canonical design lives in `00` through `10`.
