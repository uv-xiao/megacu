# Stable Docs Recreation Plan

Status: active PR closeout plan. Do not apply this directly to `docs/design/`
until the PR is ready to merge.

## Goal

During PR merge, recreate the stable design documentation so it explains the
current Megacu architecture clearly to two audiences:

- users who want to use Megacu correctly;
- contributors who need to extend platform, backend, scheduler, dispatcher,
  runtime, event, and adapter components without violating the architecture.

This is not a small append to the existing stable docs. The current
`docs/design/runtime_linked_device_native_layer/` folder should be removed.
The replacement docs should be flattened directly under `docs/design/`, not
placed inside another topic folder. The stable design entry points should be
obvious from `docs/design/README.md` and from filenames directly under
`docs/design/`.

## Why Recreate Instead Of Append

The current stable folder was written around the PR #4 proof boundary. It mixes
several layers:

- the architecture correction away from compiler/materializer design;
- PR #4 proof limits;
- stale PR #4 scope notes that say MPI and Torch adapters are out of scope;
- CUDA+NVSHMEM PR #4 example details that no longer define current PR scope;
- component contracts that have since changed around runtime execution models,
  runtime loops, and EventTensor ownership;
- implementation-plan fragments that now belong in active or future work docs.

Appending another architecture file or creating another nested design folder
would make the reading path worse. Readers would need to know which stable file
is current, which file is historical PR scope, and which component names are
superseded. The merge closeout should instead create one coherent, flat stable
design package under `docs/design/`.

## Required Stable Docs Shape

The exact filenames can change during closeout, but they should be direct
children of `docs/design/`, for example `docs/design/megacu_overview.md`,
`docs/design/programming_model.md`, and
`docs/design/runtime_components.md`. The stable package should cover these
sections in this order:

1. **Overview**
   - What Megacu is.
   - The runtime-linked architecture in one page.
   - What problem Megacu solves and what it intentionally does not solve.

2. **User Programming Model**
   - Host-called orchestrate functions.
   - Driver plus raw CUDA-kernel-like target arguments.
   - `submit(...)`, `sync(...)`, `event_tensor(...)`, and attrs.
   - What users must provide explicitly.
   - What Megacu does not infer or fallback-check.

3. **Orchestration Model**
   - The call-owned orchestration frame.
   - Task records.
   - Event tensor records.
   - Builtin task-kind attrs injected by `submit` and `sync`.
   - Event object attrs versus task event-operation attrs.

4. **Linked Components**
   - Dispatcher.
   - Scheduler.
   - Runtime execution model.
   - Runtime loop.
   - EventTensor component.
   - Platform.
   - Backend.
   - Launch adapters.
   - Driver.
   - Target call glue.

5. **Runtime Execution Models And Mega-Kernel Loops**
   - Why the component is called runtime, not entry.
   - Runtime-owned task publication/sealing model.
   - Runtime-owned device-side loop and issue policy.
   - Scheduler readiness versus dispatcher availability.
   - EventTensor issue/progress/completion hooks.
   - Candidate execution models and runtime loops, and when to use each one.

6. **Event Tensor Semantics**
   - Event Tensor as a first-class orchestration object.
   - Paper-inspired wait-count, notify, wait, and trigger semantics.
   - Common EventTensor API versus platform/backend lowering.
   - Difference from the Event Tensor paper: operator task is Megacu's sync
     granularity; finer sync requires smaller submitted tasks, not EventTensor
     subtask splitting.
   - Expressiveness comparison with the Event Tensor paper: Megacu is
     lower-level and more explicit, not less expressive. The stable docs should
     name authoring/generation burden and lowering quality as the real
     tradeoffs.
   - Sync-only tasks as native readiness tasks.
   - How event attrs lower through platform/backend code.
   - Why operators do not manually call notify, wait, trigger, backend signal,
     backend wait, or scheduler APIs.

7. **CUDA+NVSHMEM Driver And Launch Adapters**
   - Driver resource ownership.
   - Direct local launch.
   - MPI launch.
   - Torch launch.
   - Docker as the required full verification environment for MPI/Torch.

8. **Examples**
   - GEMM-RS: distributed tile-producer/tile-consumer correctness, launch
     adapters, rank-aware dispatcher/runtime behavior.
   - AG-GEMM: communication producer tasks feeding GEMM tile consumer tasks.
   - Tiny decode pipeline: non-GEMM end-to-end reuse, golden/baseline/Megacu
     comparison.
   - Why Megacu examples are composed from tile/range operator tasks, while
     handwritten large fused/persistent kernels belong only in baselines.
   - What each example proves and what it does not prove.

9. **Architecture Comparison**
   - MegaKittens/Hazy.
   - Simpler/PTO Runtime.
   - Triton-distributed.
   - What Megacu borrows and what Megacu rejects from each.

10. **Contributing Guide For Components**
    - How to add a dispatcher.
    - How to add a scheduler.
    - How to add a runtime execution model.
    - How to add a runtime loop.
    - How to add an EventTensor/backend lowering.
    - How to add a platform/backend adapter.
    - Required tests and evidence.

11. **Verification Contract**
    - Required local checks.
    - Required Docker checks.
    - Required direct/MPI/Torch checks.
    - Required example correctness checks.
    - Required no-fallback/no-inference checks.

## Migration Rules

- Remove `docs/design/runtime_linked_device_native_layer/` during PR closeout.
- Do not replace it with another nested folder under `docs/design/`; stable
  docs should be flattened directly under `docs/design/`.
- Recreate stable docs from the accepted `docs/in_progress/design/` content,
  source notes, implemented behavior, and verified examples.
- Do not copy PR-scoped historical wording into stable docs unless it remains
  true after the PR implementation.
- Reject the old distributed-runtime note that MPI and Torch adapters are out
  of scope. The current PR requires those launch adapters, with Docker as the
  full verification environment.
- Reject the old `docs/design/runtime_linked_device_native_layer/00-overview.md`
  and `06-implementation-plan.md` notes that call MPI and Torch adapters future
  work.
- Replace stale backend-specific event-namespace wording from the old stable
  component notes with the accepted EventTensor component terminology.
- Do not preserve the `entry` component name in the stable architecture. The
  linked device-side owner is `runtime`, with separate runtime execution-model
  and runtime-loop responsibilities.
- Do not preserve stale compiler/materializer terminology except in a short
  rejected-history note.
- Do not create a stable design that requires reading the old PR #4 proof docs
  to understand the new architecture.

## Closeout Checklist

- [ ] Review implemented code and examples after verification.
- [ ] Decide final flat stable docs filenames and order under `docs/design/`.
- [ ] Remove old `docs/design/runtime_linked_device_native_layer/`.
- [ ] Write the new flat stable design package from accepted in-progress docs.
- [ ] Update `docs/design/README.md`.
- [ ] Update repository `README.md` if it points to old design paths.
- [ ] Update `docs/todo/` so future work reflects what remains.
- [ ] Remove completed `docs/in_progress/` task/design drafts.
- [ ] Run `git diff --check`.
- [ ] Run the documentation policy grep used by the PR.
