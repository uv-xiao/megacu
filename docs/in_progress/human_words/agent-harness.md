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

- 2026-04-25 Asia/Shanghai - Example assets mirror platform/backend/example layout
  > Let's give a rule about examples/ folder organization. First, we need examples/<platform>-<backend>/<example> structure. That is, we need examples/cuda_nvshmem/gemm_allreduce. This should also be applied to docker/ and tools/ . And for every example, it should provide a readme to explain the example details (with visulaization and simple pseudocode) as well as usage or running scripts. This rule should be added for agent and also specified by README.md under each folder (like examples/).
  - Context: User requested a persistent organization rule while PR #3 already
    had a CUDA/NVSHMEM GEMM+AllReduce example, Docker image, and run script.
  - Related: `.agents/rules/example-organization.md`,
    `examples/README.md`, `docker/README.md`, `tools/README.md`.
  - Agent interpretation: Example code, Docker support, and scripts should be
    organized as matching platform/backend/example groups. Each example must be
    self-explaining through README files, including visualization, pseudocode,
    usage, and runtime assumptions.

- 2026-04-25 Asia/Shanghai - Add PR-comment fixing skill
  > I've glanced over the PR. You should read my comments. I think there are large problems about feature implementation (don't see any dispatch/schedule/etc., don't think we've implemented a working version) and file organization. You should look at research/repos/simpler/.claude/skills/fix-pr/ to $skill-creator create a skill about get and fix PR comments, and then rethink about docs/in_progress/public_builder_surface.md, making it a full working slice (aligned with the complete implementation-ready design).
  - Context: User reviewed PR #3 and asked for a reusable agent workflow for
    fetching and fixing PR comments, adapted from the Simpler fix-pr skill.
  - Related: `.agents/skills/fix-pr-comments/SKILL.md`, PR #3.
  - Agent interpretation: Megacu agents need a local skill that fetches PR
    feedback from GitHub, classifies it, verifies it against repo reality, fixes
    implementation/design gaps, and records blocked reply/resolve actions when
    GitHub permissions are unavailable.

- 2026-04-25 Asia/Shanghai - PR comments can be pending review comments
  > Why we need a Dockerfile under a specific example? Why not let docker/cuda_nvshmem to have only one Dockerfile
  >
  > Same problem, we shouldn't give tool/script for every example
  - Context: These comments were discovered under a pending GitHub review on PR
    #3 rather than through the normal submitted-comments endpoints.
  - Related: `.agents/skills/fix-pr-comments/SKILL.md`,
    `.agents/rules/example-organization.md`, `docker/README.md`,
    `tools/README.md`.
  - Agent interpretation: The PR-comment workflow must query pending reviews
    explicitly. The example-organization rule should prefer shared
    platform/backend Docker and tool assets, adding per-example assets only when
    a specific example truly needs unique operational support.

- 2026-04-25 Asia/Shanghai - Keep GEMM+AllReduce variants under one example family
  - Original wording partially removed from active docs because it described a
    superseded example variant that current docs must not carry forward.
  - Context: User refined the previous split example layout while PR #3 was
    being reorganized after review feedback.
  - Related: `.agents/rules/example-organization.md`,
    `examples/cuda_nvshmem/gemm_allreduce/`.
  - Agent interpretation: CUDA+NVSHMEM GEMM+AllReduce examples should keep
    shared descriptors and golden code at the family root, while baseline and
    Megacu implementation variants live below the family when active.
