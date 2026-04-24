# Feature Task: Implementation-Ready Device-Native Design

- Branch: `design/implementation-ready-device-native-layer`
- PR:
- Owner: Codex
- Status: Active

## Goal

Turn the picked Megacu device-native direction into an implementation-ready
design without treating unfinished architecture as stable implemented behavior.

## Input

- Accepted direction in `docs/design/megacu_cpp_cuda_layer.md`
- Active draft chapters under
  `docs/in_progress/design/implementation_ready_device_native_layer/`
- Human design instructions in `docs/in_progress/human_words/megakernel-design.md`
- Source-reading notes under `docs/notes/`

## Output

- A concrete active design draft with stable contracts, examples, failure modes,
  planned code paths, and verification evidence.
- `docs/design/` kept as the concise picked-direction snapshot until final PR
  merge cleanup.
- Final closeout path that merges accepted content back into `docs/design/`
  and removes stale in-progress drafts.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Move unfinished detailed design chapters out of `docs/design/`
- [ ] Make event, task, schedule, kernel, shared, platform, and backend
  contracts implementation-ready
- [x] Map initial concrete design examples to tests, generated-code inspection, or
  explicit manual evidence
- [ ] Keep `docs/design/` untouched while refining
  `docs/in_progress/design/`
- [ ] Sync `docs/design/`, `docs/todo/`, and `docs/in_progress/` at PR closeout

## Verification

- Documentation lifecycle check:
  `find docs/design docs/in_progress/design -maxdepth 2 -type f | sort`
- Path reference check:
  `rg "docs/design/megacu_device_native_layer|docs/in_progress/design/megacu_device_native_layer" docs README.md .agents -g "!docs/in_progress/human_words/**" -g "!docs/in_progress/implementation_ready_device_native_design.md"`
- Focused reread of `docs/design/README.md`,
  `docs/in_progress/README.md`, and
  `docs/in_progress/design/README.md`.

## Tests

No automated tests are required for this documentation-only PR slice.

## Docs

- Stable picked direction: `docs/design/megacu_cpp_cuda_layer.md`
- Active draft:
  `docs/in_progress/design/implementation_ready_device_native_layer/`

## Closeout

Before this PR closes, accepted implementation-ready content must be merged
back into the stable `docs/design/` layout, stale in-progress drafts must be
removed, and `docs/todo/README.md` must reflect the remaining implementation
gaps.
