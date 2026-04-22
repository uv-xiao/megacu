# Feature Task: Megacu C++/CUDA Layer Design

- Branch: `main`
- PR:
- Owner: Codex
- Status: in progress

## Goal

Design the initial Megacu C++/CUDA layer for CUDA-native megakernel creation
with explicit task, event, and scheduling concepts and no hidden abstraction
overhead on the hot path.

## Input

- User direction: build the repo harness before starting true Megacu design.
- User direction: multi-GPU support must be included from the beginning.
- `docs/notes/megakernel_cuda_layer_sources.md`
- `.agents/rules/performance-and-cuda.md`
- Existing implemented harness design in `docs/design/agent_harness.md`

## Output

- In-progress design draft under `docs/in_progress/design/`
- Concrete API and scheduling examples
- Explicit contracts, invariants, failure modes, and verification mapping
- Scope boundaries for the first executable slice

## Scope Checklist

- [x] Define input, output, and verification criteria
- [ ] Ask design-scope questions
- [ ] Present two or three viable approaches with tradeoffs
- [ ] Select one approach with user approval
- [ ] Fill in the design draft
- [ ] Map design examples to tests, benchmarks, profiler evidence, or generated-code inspection
- [ ] Verify docs locally

## Verification

- `git status --short --branch`
- `find docs/in_progress docs/notes docs/design -maxdepth 3 -type f | sort`
- focused reread of the accepted design draft

## Tests

No code tests yet. The design must define future tests and evidence before
implementation starts.

## Docs

- `docs/in_progress/design/megacu_cpp_cuda_layer.md`
- `docs/notes/megakernel_cuda_layer_sources.md`

## Accepted Constraints

- Multi-GPU concepts are part of the initial design. The first executable slice
  may be small, but task ids, event ids, memory spaces, launch model, and
  verification strategy must not assume a single GPU-only architecture.

## Closeout

After design approval, commit the task and design draft. Implementation must
start from a written plan, not directly from this task file.
