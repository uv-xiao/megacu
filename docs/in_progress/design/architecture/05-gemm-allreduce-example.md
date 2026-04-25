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

The two Megacu implementations share one `ConfigureTarget` dispatcher. The
example supplies GEMM+AllReduce-specific virtual-participant annotations to that
general dispatcher; it does not link a GEMM-specific dispatcher component.

For this example, dispatcher responsibility is visible and testable:

- single-card run: map the GEMM producer and reduction consumer to local work;
  no remote NVSHMEM peer work is required;
- two-card run: map the same participants to local tile work plus peer
  reduction work using the runtime NVSHMEM team;
- phased target: expose tile readiness so the scheduler can run reduction work
  as soon as each tile is ready;
- overlap target: emit co-residency constraints for producer/consumer
  participants when communication may block on device-side NVSHMEM waits.

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

The dispatcher participates by mapping the producer and consumer into
co-resident lane groups and recording that requirement in `dispatch_state`.
The scheduler owns the final legality check because it knows whether the linked
overlap scheduler/operator and CUDA launch envelope can keep those lane groups
live together.

## Runtime Components In The Example

The Megacu orchestrate wrapper should call runtime components in a visible
order:

```text
validate target capability envelope
validate CUDA launch and NVSHMEM team
validate workspace and symmetric storage
declare GEMM producer and reduction consumer participant attributes
create runtime dispatch state from participant attributes, problem, and team
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

constexpr auto participants = gemm_ar_participants(
    megacu::progress_requirement::nonblocking);

megacu::dispatch_state dispatch;
MEGACU_TRY(megacu::dispatcher::map(
    dispatch,
    megacu::dispatch_request{
        .context = ctx,
        .problem = megacu::problem_view::from(problem),
        .participants = participants}));

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

constexpr auto participants = gemm_ar_participants(
    megacu::progress_requirement::requires_co_resident_progress);

megacu::dispatch_state dispatch;
MEGACU_TRY(megacu::dispatcher::map(
    dispatch,
    megacu::dispatch_request{
        .context = ctx,
        .problem = megacu::problem_view::from(problem),
        .participants = participants}));

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

`gemm_ar_participants` is a small constexpr helper owned by the example:

```cpp
constexpr auto gemm_ar_participants(megacu::progress_requirement comm_progress) {
  return megacu::participant_list{
      megacu::participant<gemm_producer_tag>(
          "gemm_producer",
          {.role = megacu::participant_role::compute,
           .placement = megacu::placement_scope::per_rank,
           .progress = megacu::progress_requirement::nonblocking,
           .peer_policy = megacu::peer_policy::local_then_remote,
           .requires = megacu::participant_requirement::none}),
      megacu::participant<reduce_consumer_tag>(
          "reduce_consumer",
          {.role = megacu::participant_role::communication,
           .placement = megacu::placement_scope::per_peer,
           .progress = comm_progress,
           .peer_policy = megacu::peer_policy::all_remote_peers,
           .requires = megacu::participant_requirement::
               co_resident_progress_if_blocking})};
}
```

The dispatcher decides how those participants map to local workers and backend
peers for `team_n_pes == 1` and `team_n_pes == 2`. The CMake configuration does
not contain participant-to-rank mappings.

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
