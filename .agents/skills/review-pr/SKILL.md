---
name: review-pr
description: Perform a read-only PR or branch review using Megacu contract, performance, and verification checks.
---

# Review PR

Use this for read-only review of a branch or pull request.

## Workflow

1. Resolve the branch or PR and list changed files.
2. Classify the touched contracts: public C++/CUDA surface, platform/backend
   paths, benchmarks, docs lifecycle, and verification evidence.
3. Consult the relevant reviewer profiles under `.agents/agents/`:
   - `repo-reviewer.md`
   - `programming-surface-reviewer.md`
   - `cuda-performance-reviewer.md` when hot paths or benchmarks changed
   - `verifier.md`
4. Check docs, tests, benchmarks, and policy coverage for the touched
   contracts.
5. Report findings by severity with file paths, impact, and fix direction.

## Output

Use `.agents/templates/review-report.md`.

## Hard Rules

- Do not edit files.
- Do not change GitHub state.
- Do not run mutating build or install commands during review.
- State skipped checks and residual risk explicitly.
