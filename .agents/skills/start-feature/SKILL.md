---
name: start-feature
description: Start a Megacu feature with task and design docs before implementation.
---

# Start Feature

Use this when beginning any feature-sized change.

## Workflow

1. Confirm branch and worktree state with `git status --short --branch`.
2. Read `AGENTS.md`, `.agents/README.md`, and `.agents/rules/*.md`.
3. Choose or add a feature-sized gap in `docs/todo/README.md`.
4. Create `docs/in_progress/<feature>.md` from
   `.agents/templates/feature-task.md`.
5. If architecture, API, task/event representation, scheduling, artifacts, or
   extension boundaries change, create a design draft under
   `docs/in_progress/design/` from `.agents/templates/design.md`.
6. Commit task and design docs before implementation when git history is being
   used for the task.

## Stop Conditions

- Stop if the task is too broad to verify.
- Stop if no verification contract can be stated.
- Stop if the user asks to bypass the task/design-first workflow.

