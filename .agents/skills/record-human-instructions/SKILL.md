---
name: record-human-instructions
description: Record human instructions into docs/in_progress/human_words/ with category grouping and chronological ordering.
---

# Record Human Instructions

## Goal

Preserve source human wording during active work without turning every statement
into a project rule. Record exact instructions under
`docs/in_progress/human_words/`, grouped by category and ordered by timeline;
promote only curated decisions into rules, task files, or design docs.

## Workflow

1. Find the repository root with `git rev-parse --show-toplevel` when available.
2. Read `docs/in_progress/human_words/README.md` if present, then inspect
   existing category files with `find docs/in_progress/human_words -maxdepth 1
   -type f | sort`.
3. Choose the narrowest useful category. Prefer existing category names;
   otherwise use clear names such as `Agent Harness`, `Megakernel Design`,
   `CUDA API`, `Scheduling`, `Verification`, or `Other`.
4. Preserve the user's wording exactly or as close as the transcript allows. Put
   any agent interpretation in a separate field.
5. Record entries in chronological order inside the category file.
6. Verify by rereading the changed file and checking that the category, date,
   context, related docs, and exact wording are present.

Documentation-only recording does not require automated tests. Use a focused
reread, path check, or repo policy check when available.

