# In-Progress Design Docs

This directory holds active design work that is not ready for long-term
architecture docs yet.

Use a short top-level entry point for each workstream, then keep the detailed
chapters in one ordered subdirectory when the topic has enough depth to need
multiple files.

## Active Workstreams

- `architecture/`: active redesign of the first
  Megacu implementation around runtime-linked dispatcher/scheduler/backend
  components instead of compiler-style IR materialization. This workstream was
  moved back from `docs/design/` on 2026-04-25 after review feedback.

## Split Scope

On 2026-04-28, PR #4 was narrowed to architecture redesign plus one tiny but
mighty example. The broader concrete implementation notes moved to
`docs/todo/concrete_impl/` and are no longer part of active PR #4 scope.
