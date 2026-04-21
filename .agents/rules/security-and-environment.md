# Security And Environment Rules

- Do not commit secrets, credentials, tokens, private endpoints, or
  machine-specific absolute paths.
- Use project-relative paths in docs and scripts.
- Do not commit `research/`, `.references/`, `.repositories/`, virtual
  environments, caches, generated packages, local build output, PTX/cubin/fatbin
  artifacts, or profiler traces unless a design explicitly requires a small
  checked-in fixture.
- Ask before adding dependencies or changing CI/runtime environment
  assumptions.
- If hardware, credentials, or external services are unavailable, skip
  explicitly with a documented reason rather than faking success.

