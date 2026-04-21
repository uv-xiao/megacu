# Agent Harness Design

## Goal

Megacu starts with a repo-local human-agent collaboration harness before the
true C++/CUDA megakernel design. The harness makes agent behavior explicit,
keeps source readings separate from durable decisions, and requires verification
evidence before completion or performance claims.

## Implemented Structure

```text
AGENTS.md
  |
  v
.agents/README.md
  |
  +--> .agents/rules/      mandatory project rules
  +--> .agents/skills/     executable workflows
  +--> .agents/agents/     expert review profiles
  `--> .agents/templates/  reusable document shapes

docs/
  |
  +--> notes/        source readings and extracted lessons
  +--> in_progress/  active task files, design drafts, human wording
  +--> todo/         future or partial work
  `--> design/       implemented behavior only
```

## Contracts

- `AGENTS.md` is the concise entrypoint for agents.
- `.agents/rules/` contains durable project rules.
- `.agents/skills/` contains repeatable workflows.
- `.agents/agents/` contains read-only expert checklists unless the user asks
  for delegated subagent work.
- `.agents/templates/` contains reusable task, design, PR, and review shapes.
- `docs/notes/` contains curated reading reports; local source material stays
  under ignored directories.
- `docs/design/` contains implemented behavior only.
- `docs/todo/` contains future or partial work only.
- `docs/in_progress/` contains active task files, draft designs, and active
  human wording.
- `research/`, `.references/`, and `.repositories/` are ignored local input
  directories.

## Workflow

1. Agents read `AGENTS.md`, `.agents/README.md`, and all rules before editing.
2. Feature-sized work starts from a `docs/todo/` gap and a task file under
   `docs/in_progress/`.
3. Architecture, API, task/event representation, scheduling, or artifact changes
   use the `design-first` skill and create a draft under
   `docs/in_progress/design/`.
4. Substantial source readings are recorded under `docs/notes/`.
5. Human wording that affects active work is preserved under
   `docs/in_progress/human_words/` before curated decisions are promoted.
6. Completion and performance claims require fresh verification evidence.

## Megacu-Specific Rules

Performance-sensitive work follows `.agents/rules/performance-and-cuda.md`.
The important project-specific distinction is that "zero overhead" never means
"synchronization is free." It means the abstraction emits the CUDA operations an
expert would intentionally write by hand, with explicit costs for atomics,
queues, waits, polling, and remote signaling.

## Verification

The initial scaffold was verified with:

```bash
git status --short --branch
find .agents docs -maxdepth 3 -type f | sort
git check-ignore research/papers/mpk-2512.22219.pdf
git check-ignore research/repos/mirage-mpk/README.md
```

