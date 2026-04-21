---
name: design-first
description: Turn an architectural or API request into a recorded Megacu design before implementation.
---

# Design First

Use this when work changes architecture, CUDA programming surface, task/event
representation, scheduling policy, generated artifacts, or extension seams.

## Workflow

1. Gather context from `README.md`, `docs/design/`, `docs/todo/`,
   `docs/in_progress/`, and relevant `docs/notes/`.
2. Record source readings in `docs/notes/` when external papers, repositories,
   benchmarks, or CUDA documentation affect the design.
3. Present two or three viable approaches with tradeoffs.
4. Select one approach and record it under `docs/in_progress/design/`.
5. Include concrete examples during the design phase. Examples may be small, but
   each example must show user-visible API, task/event behavior, or scheduling
   behavior.
6. Convert each design example into tests, benchmarks, profiler evidence, or
   generated-code inspection criteria. If an example cannot become automated
   yet, name the manual evidence that will prove it.
7. Define contracts, invariants, failure modes, and verification evidence.
8. Do not implement until the design has clear acceptance criteria.

## Design Quality Bar

- Explain why the design exists.
- List concrete contracts and owners.
- Show feature behavior with concrete examples before implementation starts.
- Include planned code paths or artifact paths.
- Define how examples become tests or evidence.
- State what is intentionally out of scope.

