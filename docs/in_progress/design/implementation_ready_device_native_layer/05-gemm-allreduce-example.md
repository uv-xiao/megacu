# GEMM+AllReduce Example

The CUDA+NVSHMEM GEMM+AllReduce example is the first proof target for the
runtime-linked design.

Required layout:

```text
examples/cuda_nvshmem/gemm_allreduce/
  common/
  golden/
  phased/baseline/
  phased/megacu/
  overlap/baseline/
  overlap/megacu/
```

## Four Native Baselines And Two Megacu Paths

Golden and baseline roles must be separated:

```text
golden:
  local expected GEMM results

phased/baseline:
  pure CUDA/NVSHMEM phased implementation, single-card and two-card

overlap/baseline:
  pure CUDA/NVSHMEM overlap implementation, single-card and two-card

phased/megacu:
  direct Megacu orchestrate target linked with runtime dispatcher/scheduler

overlap/megacu:
  direct Megacu orchestrate target linked with runtime dispatcher/scheduler
```

There should be two Megacu implementations, not four. Each Megacu target uses
runtime team/backend views to support single-card and two-card cases.

## Phased Runtime Path

Phased does not mean a fake whole-program dependency. It means communication
does not rely on co-resident blocking progress. The scheduler may still start a
reduction tile as soon as its input tile is ready.

```text
for tile in runtime dispatcher:
  launch/execute GEMM tile
  mark tile ready
  when tile ready and backend permits:
    execute AR tile
```

The first implementation may be conservative if needed, but tests should make
false whole-program dependencies visible.

## Overlap Runtime Path

Overlap requires a progress guard:

```text
persistent/co-resident execution:
  compute workers:
    produce tile
    signal tile-ready event

  communication workers:
    wait tile-ready event
    run device-side NVSHMEM reduction
```

The scheduler must reject blocking communication waits when the linked operator
or launch shape cannot prove compute and communication progress can happen
together.

## Runtime Components In The Example

The Megacu orchestrate wrapper should call runtime components in a visible
order:

```text
validate target capability envelope
validate CUDA launch and NVSHMEM team
validate workspace and symmetric storage
create runtime dispatch state from problem and team
call phased or overlap scheduler
scheduler calls linked native operator symbols
```

The example must not call a materializer, build metadata sections, or route
through golden functions as its Megacu implementation once the runtime-linked
replacement is done.

## Concrete Megacu Phased Pseudocode

```cpp
auto ctx = make_runtime_context(
    launch, team, events, cuda_nvshmem_gemm_allreduce_phased_capability());
MEGACU_TRY(validate_common(ctx, workspace, problem));

gemm_ar_dispatch dispatch;
MEGACU_TRY(prepare_gemm_ar(dispatch, problem, team, ctx.capability));

return run_phased_gemm_ar(
    ctx,
    dispatch,
    workspace,
    problem,
    operators::gemm_ar_phased{
        .run = megacu_cuda_gemm_allreduce_phased_f32});
```

## Concrete Megacu Overlap Pseudocode

```cpp
auto ctx = make_runtime_context(
    launch, team, events, cuda_nvshmem_gemm_allreduce_overlap_capability());
MEGACU_TRY(validate_common(ctx, workspace, problem));
MEGACU_TRY(validate_overlap_progress_guard(ctx.capability, launch));

gemm_ar_dispatch dispatch;
MEGACU_TRY(prepare_gemm_ar(dispatch, problem, team, ctx.capability));

return run_overlap_gemm_ar(
    ctx,
    dispatch,
    workspace,
    problem,
    operators::gemm_ar_overlap{
        .run = megacu_cuda_gemm_allreduce_overlap_f32});
```

The exact helper names can change, but the call order cannot collapse back into
metadata validation plus a direct golden function call.

## Golden And Baseline Separation

The `golden/` directory should provide expected results, not the implementation
that Megacu calls in production. Native baseline implementations should live
under `phased/baseline/` and `overlap/baseline/`. Megacu implementations should
link their own operator symbols under `phased/megacu/` and `overlap/megacu/`.

## Usage Evidence

Each variant README should include:

- what the variant proves;
- a simple path visualization;
- pseudocode;
- build command;
- single-card run command;
- two-card `nvshmrun` or Docker command;
- known unsupported shapes/features.
