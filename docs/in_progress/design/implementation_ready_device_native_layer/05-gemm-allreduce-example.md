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

## Usage Evidence

Each variant README should include:

- what the variant proves;
- a simple path visualization;
- pseudocode;
- build command;
- single-card run command;
- two-card `nvshmrun` or Docker command;
- known unsupported shapes/features.
