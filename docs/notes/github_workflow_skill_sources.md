# GitHub Workflow Skill Source Notes

- Date: 2026-04-24 Asia/Shanghai
- Purpose: adapt IntelliC's repo-local GitHub workflow skills into Megacu's
  harness without importing IntelliC-specific assumptions.
- Related work: Megacu agent harness refinement

## Source Scope

### IntelliC reference harness

- Local sibling repo: `../intellic`
- Files read:
  - `AGENTS.md`
  - `.agents/README.md`
  - `.agents/skills/create-pr/SKILL.md`
  - `.agents/skills/review-pr/SKILL.md`
  - `.agents/skills/clean-branches/SKILL.md`
  - `.agents/templates/pr-body.md`
  - `.agents/templates/review-report.md`
  - `docs/design/agent_harness.md`
  - `docs/in_progress/human_words/branch-management.md`

### Megacu harness context

- Files read:
  - `AGENTS.md`
  - `.agents/README.md`
  - `.agents/rules/*.md`
  - `.agents/templates/pr-body.md`
  - `.agents/templates/review-report.md`
  - `.agents/agents/{repo-reviewer.md,programming-surface-reviewer.md,cuda-performance-reviewer.md,verifier.md}`

## What IntelliC Contributes

IntelliC already has a clean repo-local pattern for GitHub-facing workflows:
small `SKILL.md` files, no extra metadata files, and direct references to local
templates and reviewer profiles. The useful parts are:

- `create-pr` keeps publishing narrow: verify, sync docs, write the PR body,
  then push and create or update the PR.
- `review-pr` is explicitly read-only and routes judgment through local reviewer
  profiles instead of trying to embed every review rule directly in the skill.
- `clean-branches` uses a careful discover-classify-delete flow and requires
  explicit approval before deletion.

## Megacu-Specific Tuning

Megacu needed different guardrails from IntelliC:

- docs lifecycle checks must include `docs/notes/` because source-reading notes
  are part of Megacu's policy surface;
- PR publication must block unsubstantiated performance or zero-overhead claims;
- review should classify CUDA hot paths, public C++/CUDA programming surface,
  verification evidence, and docs state, then route to the matching Megacu
  reviewer profiles;
- branch cleanup can keep the IntelliC safety model largely unchanged because it
  is repository-agnostic.

## Affected Decisions

- Add repo-local `create-pr`, `review-pr`, and `clean-branches` skills under
  `.agents/skills/`.
- Reuse Megacu's existing PR body and review report templates rather than
  importing IntelliC template files.
- Keep the skills concise and repo-local; do not add a parallel `.codex` or
  plugin-specific workflow layer.

## Follow-Up Verification

- YAML frontmatter parse check for each new `SKILL.md`
- path check for referenced templates, reviewer profiles, and docs directories
- `git diff --check`
