# Human Words: Agent Harness

## Category

- Primary: Agent Harness

## Historical Path Note

Older `Related:` paths below may point at `docs/in_progress/` locations that
were later promoted into `docs/design/`.

## Timeline

- 2026-04-21 Asia/Shanghai - Harness before Megacu design
  > repo harness should be built before starting true megacu design
  - Context: User clarified priority after reviewing the MPK/Event Tensor
    research note.
  - Related: `docs/in_progress/agent_harness.md`, `.agents/rules/`,
    `docs/todo/README.md`
  - Agent interpretation: Build the agent-friendly repository workflow before
    beginning the true C++/CUDA layer design.

- 2026-04-21 Asia/Shanghai - Start Megacu design after harness
  > Then, let's start the megacu's design
  - Context: User requested the real design work after the research note, later
    clarifying that the repo harness must come first.
  - Related: `docs/in_progress/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`
  - Agent interpretation: Start a design-first workflow for the C++/CUDA layer
    after committing the initial harness.

- 2026-04-24 01:32 CST - Add GitHub workflow skills from IntelliC into Megacu
  > You should look at ~/intellic to create the skills to work with github, including create-pr. $skill-creator
  - Context: User requested repo-local GitHub workflow skills by adapting the
    IntelliC harness patterns.
  - Related: `.agents/skills/`, `.agents/templates/pr-body.md`
  - Agent interpretation: Use IntelliC as the reference harness, but land the
    resulting GitHub workflow skills inside Megacu's repo-local agent harness.

- 2026-04-24 01:32 CST - Broaden the import and tune it for Megacu
  > into megucu
  > Import more and tuned.
  - Context: User corrected the destination repo and rejected a minimal
    create-pr/review-pr-only port.
  - Related: `.agents/skills/create-pr/SKILL.md`,
    `.agents/skills/review-pr/SKILL.md`,
    `.agents/skills/clean-branches/SKILL.md`
  - Agent interpretation: Import the broader GitHub workflow set and adapt the
    workflow and guardrails to Megacu's docs lifecycle, reviewer profiles, and
    verification expectations.

- 2026-04-24 01:32 CST - Commit, push, and refresh PR description
  > commit and push to the PRhttps://github.com/uv-xiao/megacu/pull/1, also, use the create-pr skills to re-edit the PR description.
  - Context: User asked to publish the new GitHub workflow skills on the
    existing branch PR and refresh the PR body through the repo-local workflow.
  - Related: `.agents/skills/create-pr/SKILL.md`,
    `.agents/templates/pr-body.md`,
    `https://github.com/uv-xiao/megacu/pull/1`
  - Agent interpretation: Commit only the new harness-skill slice, push it to
    `story-level-design-pr`, and rewrite PR #1's description using Megacu's
    PR-body template.

- 2026-04-24 01:32 CST - Amend the PR commit to include the pending harness refinements
  > These should also be commited (ammend) and pushed
  - Context: User asked to include the remaining local harness-rule and
    recording-skill edits in the existing PR branch via amend.
  - Related: `.agents/rules/docs-and-knowledge.md`,
    `.agents/rules/performance-and-cuda.md`,
    `.agents/skills/record-human-instructions/SKILL.md`
  - Agent interpretation: Amend the last pushed PR commit to include the
    pending rule and skill refinements, then push the rewritten branch state.

- 2026-04-24 01:32 CST - Require design promotion before PR merge
  > We should give a new agent rule (commited by amend), which requires that before PR merging, in_progress documents should be moved and merged into docs/design (not only move, but should be unifiedly merged, which might need re-organization of the docs/design; you also need to update README.md if necessary). And you should do as the rule suggests before the PR merging.
  - Context: User corrected the workflow after PR #1 had already been merged
    and then asked for the accepted design to be promoted directly on `main`.
  - Related: `.agents/rules/development-flow.md`,
    `.agents/skills/create-pr/SKILL.md`,
    `docs/design/`,
    `README.md`
  - Agent interpretation: Add a normative pre-merge rule that accepted design
    content must be unified into `docs/design/` with updated indexes and
    README entry points, and apply that promotion directly on `main` for the
    accepted Megacu design.
