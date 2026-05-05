# Tiny Phased GEMM+AllReduce Proof Example

The PR #4 example is intentionally tiny and phased-only. Its job is to prove
the runtime-linked architecture path for 1-host/1-device and 1-host/2-device,
not to become the general CUDA+NVSHMEM implementation.

The example may be problem-specific. It may use one fixed dtype and one small
shape family. The dispatcher, scheduler, backend event tensor, CUDA
mega-kernel entry, and device running loop should still be common linked
components. A handwritten monolithic mega-kernel is useful as a baseline, but
it must live in the baseline directory rather than the Megacu path. Those
shortcuts must stay out of public Megacu APIs and shared component contracts.

## What Makes It Mighty

Even though the example is tiny, it should cross the important architectural
boundaries:

```text
target orchestrate entry with direct target args
  -> target runtime receives driver resources and direct target arguments
  -> authoring code submits GEMM and phased AR tasks
  -> task dependencies are read from explicit dependency attributes
  -> dispatcher maps work for 1-device or 2-device
  -> ASAP scheduler consumes explicit dependency readiness
  -> CUDA/NVSHMEM resources are consumed by platform/backend components
  -> status returns to caller
```

The example proves that Megacu is ordinary linked C++/CUDA runtime code with a
programmable authoring surface. It must not prove MPI, torch-distributed,
arbitrary team sizes, retired communication-progress variants, or the final
general attribute system.
It also clarifies the PR #4 meaning of one Megacu megakernel: one host
orchestrate call can dispatch and schedule the linked native GEMM and phased
all-reduce kernels internally.

The PR #4 implementation takes the thinnest conforming path: the authoring code
still submits GEMM, sync-only readiness, and AllReduce tasks, but the target
runtime validates that task graph and lowers it to one native CUDA+NVSHMEM
launcher. That launcher instantiates linked device components: common
device-entry running logic, explicit ASAP scheduler, tile-grid dispatcher,
CUDA platform mega-kernel shell, NVSHMEM backend event tensor, and a small
operator table. This avoids both bad extremes: a misleading host sequence where
each submitted operator is launched separately, and an example-authored manual
fused mega-kernel pretending to be Megacu lowering.

## Minimal Target Shape

The workload entry is a host-called function. It receives a CUDA+NVSHMEM driver
for asynchronous and distributed execution resources, then direct target
arguments shaped like CUDA kernel arguments:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver_view driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem);
```

The problem can be narrow:

- f32 row-major input and output;
- one local GEMM tile or a tiny fixed tile grid;
- 1-host/1-device and 1-host/2-device;
- phased execution only;
- small linked operator task bodies composed by the Megacu CUDA+NVSHMEM
  lowering path.

## Authoring Sketch

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver_view driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem) {
  auto phase = megacu::runtime::make_phase(
      driver,
      megacu::runtime::progress_model::asap);

  auto gemm = phase.submit(
      gemm_tile_f32,
      a, b, partial, problem.m, problem.n, problem.k,
      attrs(dispatcher::tile_grid(problem.m, problem.n)));

  phase.submit(
      phased_allreduce_f32,
      partial, out, problem.m, problem.n,
      attrs(scheduler::depends_on(gemm)));

  return phase.run();
}
```

The sketch is intentionally not dispatcher/scheduler boilerplate. The linked
target runtime turns the submitted tasks and explicit attributes into component
calls.

## Concrete Tiny Call Path

```text
test/native driver
  -> build CUDA+NVSHMEM driver with stream/device/team resources
  -> call gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, events, problem)
  -> authoring code declares an event tensor over the events storage
  -> authoring code submits GEMM task, sync-only readiness task, and AR task
  -> dependency model records explicit GEMM -> sync -> AR attributes
  -> event attrs record publish -> join -> acquire handlers
  -> dispatcher maps work:
       1-device: local work only
       2-device: local work plus peer reduction work
  -> Megacu validates the graph and invokes one fused phased native launcher
  -> launcher starts one mega-kernel
  -> backend lowers event attrs to local or CUDA+NVSHMEM synchronization
  -> status returns
```

## Golden, Baseline, And Megacu Roles

For PR #4:

- `golden`: local expected result generation for the tiny shape;
- `baseline`: optional handwritten phased CUDA/NVSHMEM comparison only if it
  clarifies validation;
- `megacu`: the direct runtime-linked phased path.

The full future suite remains TODO work:

```text
examples/cuda_nvshmem/gemm_allreduce/
  common/
  golden/
  phased/baseline/
  phased/megacu/
```

## Rejected Path

The tiny example must not take this path:

```text
program_builder
  -> program_ir
  -> materialize_program
  -> dispatch_section/schedule_section/kernel_section/backend_section
  -> generated or materialized target metadata
  -> example glue selects native symbol
```

Those steps are the architecture being rejected.

## Evidence Expected From The Tiny Example

The tiny example should provide evidence for:

- target arguments are configurable through a direct target signature, not fixed
  platform parameters or a generic runtime frame;
- CUDA/NVSHMEM resources are driver resources, not target arguments;
- task arguments are passed raw, without `input`/`output`/`inout` wrappers;
- task dependencies are represented through explicit dependency attributes, not
  inferred from argument declarations or pointer aliasing;
- missing dependencies are the target author's responsibility; Megacu does not
  fallback-check raw arguments to infer omitted edges;
- the linked native operator path is called without a materializer;
- 1-host/1-device phased correctness;
- 1-host/2-device phased correctness using device-side NVSHMEM.
