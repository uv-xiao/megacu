# Megacu Agent Guide

This file is the entrypoint for agents working in Megacu.

## Read First

Before editing, agents must:

1. run `git status --short --branch`
2. read `.agents/README.md`
3. read all files under `.agents/rules/`
4. load any task-relevant skill under `.agents/skills/<name>/SKILL.md`
5. inspect relevant docs under `docs/design/`, `docs/todo/`,
   `docs/in_progress/`, and `docs/notes/`

## Harness Layout

- `.agents/rules/`: persistent project rules
- `.agents/skills/`: executable workflows
- `.agents/agents/`: expert consultation profiles
- `.agents/templates/`: reusable task, design, PR, and review templates

Keep local research inputs under ignored directories such as `research/`,
`.references/`, and `.repositories/`. Commit curated notes, designs, tasks, and
rules instead.

