---
name: create-pr
description: Prepare, push, and create or update a Megacu pull request after local verification.
---

# Create PR

Use this when publishing a Megacu feature branch.

## Workflow

1. Confirm the current branch is not `main` and capture
   `git status --short --branch`.
2. Identify the active task and design docs that define the branch scope.
3. Run the relevant verification skill and any task-specific checks.
4. Ensure the docs lifecycle is consistent with the branch state:
   - `docs/design/` contains implemented behavior only
   - `docs/todo/` contains future or partial work only
   - `docs/in_progress/` contains active tasks and active design drafts only
   - `docs/notes/` contains any source-reading notes that affected the branch
   - if the branch closes a design track, accepted design content has already
     been merged into the current `docs/design/` structure and stable README
     entry points are updated
5. Write a PR body using `.agents/templates/pr-body.md`.
6. Push the branch.
7. Create or update the PR with GitHub tooling, carrying over the task,
   verification evidence, and any benchmark context.

## Guardrails

- Never force-push without explicit user approval.
- Never publish from `main`.
- Never publish with stale completed design drafts under
  `docs/in_progress/design/`.
- Never ask for merge while accepted design content still lives only under
  `docs/in_progress/`.
- Never publish performance or zero-overhead claims without benchmark or source
  evidence.
- Never include AI co-author lines.
