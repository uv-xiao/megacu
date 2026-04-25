---
name: fix-pr-comments
description: Fetch, classify, and fix GitHub PR comments or review feedback for Megacu PRs, including CI checks and design-contract alignment.
---

# Fix PR Comments

Use this when a Megacu PR has review comments, unresolved review threads, CI
failures, or human feedback saying the PR does not match the design.

## Workflow

1. Resolve the PR.
   - Prefer the explicit PR number from the user.
   - Otherwise inspect the current branch and `gh pr status`.
   - Confirm the PR is open before changing the branch.
2. Fetch feedback from all available surfaces.
   - GitHub connector: PR metadata, top-level comments, reviews, review
     threads, changed files, and CI status.
   - `gh` fallback:
     - `gh api repos/<owner>/<repo>/issues/<pr>/comments --paginate`
     - `gh api repos/<owner>/<repo>/pulls/<pr>/comments --paginate`
     - GraphQL `reviewThreads(first: 100)` for unresolved inline threads.
     - Pending reviews:
       `gh api repos/<owner>/<repo>/pulls/<pr>/reviews`, then
       `gh api repos/<owner>/<repo>/pulls/<pr>/reviews/<review-id>/comments`
       for any `PENDING` review.
     - `gh pr checks <pr>` for checks.
   - If no GitHub comments are visible, state that and use the user's message
     as the actionable feedback source.
3. Classify every item before editing.
   - **Blocking:** incorrect behavior, missing implementation slice, failing CI,
     broken verification, or design-contract mismatch.
   - **Architecture:** file organization, component boundaries, API thinness,
     CMake/runtime ownership, backend/platform capability range.
   - **Docs:** task/design/README claims that are stale, overbroad, or missing
     implementation details.
   - **Optional/discussable:** style or scope requests that may conflict with
     existing Megacu rules.
4. Verify against the repo before accepting feedback blindly.
   - Read the affected files and the relevant docs under `docs/design/`,
     `docs/in_progress/`, `docs/todo/`, and `.agents/rules/`.
   - For implementation feedback, check whether real dispatcher, scheduler,
     kernel-lowering, platform, backend, and runtime paths exist, not just
     metadata labels or example-local shortcuts.
   - Push back only with concrete evidence from code, tests, or accepted design.
5. Fix in small batches.
   - Keep unrelated user changes intact.
   - Update task docs when the review changes scope or exposes overclaims.
   - Implementation fixes must include tests or a recorded verification reason.
   - Documentation-only fixes must be reread and checked for stale claims.
6. Re-check.
   - Run focused tests for changed contracts.
   - Run broader CMake/CTest or example validation when implementation changed.
   - Run `git diff --check` before handoff.
7. Reply and resolve only when appropriate.
   - If the tool token has permission, reply in the specific review thread and
     resolve it after the fix is committed or clearly documented.
   - If permissions fail, record the exact failure and include the intended
     reply in the handoff.
   - Do not resolve comments that were skipped, only discussed, or not actually
     fixed.

## Megacu Review Checks

For PRs claiming a working implementation slice, verify these before closeout:

- the PR does not stop at public headers, metadata counts, or example-local
  wrappers;
- dispatcher, scheduler, kernel-lowering, platform, and backend ownership is
  present in repo paths or explicitly listed as missing work;
- examples follow `examples/<platform>_<backend>/<example>/`, with matching
  `docker/` and `tools/` trees when needed;
- each example owns its `CMakeLists.txt` and README;
- CUDA/NVSHMEM examples distinguish golden native baselines from Megacu paths;
- device-side NVSHMEM requirements and config capability ranges are explicit;
- task docs do not claim completion until correctness and contract checks have
  fresh evidence.

## Output

Report:

- visible PR feedback sources and any inaccessible surfaces;
- actionable items addressed, skipped, or still open;
- files changed;
- verification commands and outcomes;
- GitHub reply/resolve actions attempted or blocked.
