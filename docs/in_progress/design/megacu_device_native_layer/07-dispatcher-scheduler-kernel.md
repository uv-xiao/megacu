# Dispatcher, Scheduler, And Kernel Lowering

This chapter separates three things that were previously too entangled.

## Dispatcher

The dispatcher has two build-graph parts:

1. a reusable compiled implementation
2. an orchestrate-target lowering step on one concrete program

Together they map logical program work to execution-level placements.

Inputs from the program model:

- named op
- logical work domain
- resource usage
- explicit event dependencies
- optional mapping hints

Dispatcher responsibilities:

- decide how logical work is partitioned
- decide placement groups or execution lanes
- produce the execution-level work units consumed by the scheduler

The dispatcher is independent from the scheduler. This matches the user's point:
mapping strategy and scheduling strategy are not the same thing.

## Scheduler

The scheduler also has two build-graph parts:

1. a reusable compiled engine
2. an orchestrate-target payload specialized for one concrete program

The scheduler consumes dispatcher output. It does not define the author's public
API.

Examples:

- static persistent schedule
- queue-backed schedule
- scheduler-CTA model
- cluster-synchronous model

## Kernel Lowering

Kernel lowering also splits into reusable compiled logic plus orchestrate-target
specialization.

It decides how named kernels are turned into executable device behavior for the
chosen scheduler.

Examples:

- keep named kernels as separate launches inside the orchestrate-controlled
  execution path
- stitch several named kernels into one persistent kernel
- generate helper glue around backend/platform interaction

This is also where fine-grained overlap needs careful treatment.

## Fine-Grained Compute/Communication Overlap

The user concern is correct: fine-grained overlap is important, and a model that
treats every op as a fully opaque launched kernel would be too weak.

The chosen solution is:

- keep a single public named-op abstraction
- allow backend primitives inside named kernel bodies
- let kernel lowering stitch named kernels when the chosen target strategy
  supports it

So fine-grained overlap comes from two places:

1. in-kernel backend/platform primitives used inside named kernels
2. reusable engines plus orchestrate-target stitching/composition by the
   lowering pipeline

Megacu does not need a public fragment taxonomy to support this.

## Build-Graph Ownership

Dispatcher, scheduler, and kernel lowering implementations are compiled first,
then chosen and specialized during orchestrate-target lowering.

Runtime orchestration must not choose among them.

## Implementation Architecture

The implementation should likely organize these as separate internal components:

- dispatcher implementation
- scheduler implementation
- kernel-lowering implementation
- target-lowering driver that applies them to one concrete orchestrate program

But they are not separate public authoring layers. Their composition is part of
the CMake-managed build graph that produces one compiled orchestrate target.
