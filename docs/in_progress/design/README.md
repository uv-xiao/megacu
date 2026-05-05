# In-Progress Design Docs

This directory holds active design work that is not ready for long-term
architecture docs yet.

Use a short top-level entry point for each workstream, then keep the detailed
chapters in one ordered subdirectory when the topic has enough depth to need
multiple files.

## Active Workstreams

- `general_runtime_linked_components.md` - active design for required
  MPI/Torch launch adapters, strategy selection, reusable runtime-linked
  component contracts, and the GEMM-RS/AG-GEMM/tiny-decode example scope.
- `overall_runtime_architecture.md` - active overall architecture view for the
  orchestration frame, task/event model, runtime execution model, runtime loop,
  linked components, and comparison with related systems. Promote accepted content into
  `docs/design/` only during PR closeout.
- `execution_model_study.md` - standalone study of runtime execution models:
  who builds task records, when tasks become visible to the runtime, and how
  `host-orch`, `seeded-orch`, future device-orch, and distributed execution
  models compare.
- `runtime_execution_model_implementation_design.md` - implementation design
  for the PR-required `host-orch` and `seeded-orch` runtime execution models,
  including arena records, runtime composition, example layout, and
  verification gates.
- `stable_docs_recreation_plan.md` - closeout plan for recreating stable
  `docs/design/` documentation during PR merge, replacing the current
  `runtime_linked_device_native_layer/` folder with a clearer audience-facing
  structure.

## Split Scope

On 2026-04-28, PR #4 was narrowed to architecture redesign plus one tiny but
mighty example. The broader concrete implementation notes moved to
`docs/todo/concrete_impl/` and remain future work.
