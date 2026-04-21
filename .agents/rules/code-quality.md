# Code Quality Rules

- Write typed, explicit, human-readable code.
- Prefer small modules with clear ownership over monolithic helpers.
- Avoid stringly semantic ownership when typed ids, typed references, classes,
  structs, or enums are viable.
- Avoid dict-shaped or untyped public APIs unless the shape is an explicit
  serialization boundary.
- Keep public APIs documented with concise comments when names alone do not
  explain the contract.
- Keep implementation comments focused on invariants, non-obvious decisions,
  memory ordering, CUDA synchronization, and cross-module contracts.
- Do not add plan-specific comments such as `Phase 1` or `Step 3` inside
  production code.
- Keep examples and tests readable top-to-bottom; do not hide important state in
  global payload blobs.

