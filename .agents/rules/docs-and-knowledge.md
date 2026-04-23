# Docs And Knowledge Rules

- `docs/design/` contains implemented behavior only.
- `docs/todo/` contains future or partial work only.
- `docs/in_progress/` contains active feature tasks and active design drafts
  only.
- Record decision-bearing human instructions under
  `docs/in_progress/human_words/` during active work. Preserve the user's
  wording, date, and context; promote only curated decisions into rules, tasks,
  or design docs. Skip low-signal coordination messages such as "continue",
  "go on", acknowledgements, status pings, or thanks unless they contain a
  concrete decision or new constraint.
- `docs/notes/` contains document-reading and repository-reading reports.
- Local reading inputs under `research/`, `.references/`, and `.repositories/`
  remain ignored and uncommitted.
- When an agent reads a substantial document or repository for project
  decisions, write or update a `docs/notes/` report with source, date, purpose,
  source scope, structure/process diagrams when useful, code or API sketches,
  comparison tables when sources are weighed, extracted lessons, affected
  decisions, and follow-up verification evidence.
- `docs/notes/` reports must not be summary-only when the source affects design,
  rules, tasks, or implementation direction.
- Notes are not normative until promoted into `docs/design/`, `docs/todo/`, or
  `.agents/rules/`.
- Design docs must explain rationale, contracts, examples, planned code paths,
  failure modes, and verification evidence.
- Documentation-only changes do not need automated tests; verify them with
  reading, link/path checks, policy checks, or rendered-doc inspection as
  appropriate.
- Do not leave stale duplicates across `docs/design/`, `docs/todo/`, and
  `docs/in_progress/`.
