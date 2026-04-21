# Evidence And Verification Rules

- Every feature task needs explicit input, output, and verification criteria.
- Design drafts must include concrete examples that show feature behavior.
- Design examples must map to tests or evidence before implementation starts.
- Verify design assumptions before implementation whenever behavior can be
  tested or demonstrated.
- Use minimal reproducible demos before debugging a complex system through broad
  code search.
- Do not claim work is complete, fixed, passing, faster, or zero-overhead
  without fresh command output or benchmark evidence from the relevant
  verification.
- Prefer focused verification first, then broader checks.
- Documentation-only changes do not require automated tests, but still require
  concrete verification evidence such as a focused reread, link/path check, or
  policy check.
- Code or behavior changes must include tests or a documented reason tests are
  impossible for the current scope.
- Do not weaken tests, assertions, or policy checks to make CI pass unless the
  contract intentionally changed and docs are updated.
- Capture exact verification commands in task files, PR bodies, or handoff
  notes.

