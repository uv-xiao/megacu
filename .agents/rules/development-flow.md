# Development Flow Rules

- Work on feature-sized branches once the project has a remote or shared branch
  workflow.
- Pick or create a feature-sized gap in `docs/todo/README.md`.
- Create a task file under `docs/in_progress/` before implementation.
- Architecture-changing work must create design drafts under
  `docs/in_progress/design/` before implementation.
- Multi-file work requires a written plan with explicit input, output, and
  verification criteria.
- Before a pull request is merged, accepted design content must be promoted out
  of `docs/in_progress/` into a unified `docs/design/` layout. Do not just move
  files blindly: merge the accepted design into the current `docs/design/`
  structure, update `docs/design/README.md` and repository `README.md` entry
  points when paths or stable docs change, and remove stale in-progress drafts
  and task files.
- Before closing a task, move implemented design into `docs/design/`, update
  `docs/todo/`, remove stale task files, and remove completed
  `docs/in_progress/design/` drafts.
- Do not preserve legacy parallel systems after a redesign supersedes them.
- Commit in coherent slices that match the task plan.
