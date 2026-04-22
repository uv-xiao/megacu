# Design: Megacu C++/CUDA Layer

## Goal

Design pending. The design starts after approach selection.

## Context

- MPK and Event Tensor source reading is recorded in
  `docs/notes/megakernel_cuda_layer_sources.md`.
- Megacu's project-specific performance rules are recorded in
  `.agents/rules/performance-and-cuda.md`.
- The repo harness is implemented in `docs/design/agent_harness.md`.
- The first design must include multi-GPU concepts from the beginning. This
  affects identifiers, event ownership, memory-space modeling, launch topology,
  synchronization contracts, and verification evidence.

## Accepted Constraints

- Multi-GPU is not a later extension. The initial architecture must include
  remote event ownership and cross-GPU communication concepts, even if the first
  executable benchmark uses a narrow subset.
- Static scheduling remains the default design direction unless an explicitly
  dynamic region has a measured or clearly stated reason.
- "Zero overhead" means CUDA-native explicit costs, not free synchronization.

## Alternatives

Pending.

## Selected Design

Pending.

## Examples

- Example: Pending.
- Feature shown: Pending.
- Verification mapping: Pending.

## Contracts

Pending.

## Failure Modes

Pending.

## Verification

Pending.

## Out Of Scope

Pending.

## Closeout

Pending user review.
