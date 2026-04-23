# Positioning And Audience

## Goal

Megacu should not compete by becoming a larger compiler/runtime than MPK, a new
Event Tensor compiler, a Triton-distributed replacement, or a MoE-specific
system like UniEP. Its special role is to be the thin device-native substrate
that lets expert kernel authors compose those ideas without giving up control of
the hot path.

Coverage against those systems is a design test, not a promise to copy each
public mechanism they expose.

## Competitor Map

| System | What it owns | Strength | Gap Megacu can target |
| --- | --- | --- | --- |
| MPK | End-to-end PyTorch/compiler/runtime for model mega-kernelization | automatic LLM inference fusion, SM-level task graph, runtime scheduling | heavy vertical system; user does not directly author arbitrary native kernels as the primary API |
| Event Tensor | Compiler abstraction for tensor-shaped synchronization | concise representation of many fine-grained dependencies, symbolic/dynamic events | compiler-centric; public implementation not available; does not define a native-kernel authoring layer |
| Triton-distributed | Triton language/runtime extensions for distributed kernels | compact distributed primitives and backend lowering across SHMEM-like transports | tied to Triton programming model; not a C++ native kernel substrate |
| MegaKittens | Modular instruction metadata and fixed CUDA worker runtime | practical project organization for optimized model instructions | specialized pipeline; not a general event/backend abstraction |
| UniEP | Expert-parallel MoE training mega-kernel | focused production story: deterministic token order, scoreboard, dynamic SM roles, autotuning | vertical MoE system; useful patterns are not exposed as reusable general C++ contracts |
| FlashInfer | PyTorch-facing kernel library and generator for inference kernels | strong framework ergonomics: package split, plan/run wrappers, workspace ownership, optional side outputs, diagnostics, JIT/artifact cache | operator library, not a megakernel task/event/schedule layer; integration lessons should inform Megacu adapters without requiring a first-party Python layer |
| PTO Runtime / simpler | runtime-owned orchestration and scheduling for submitted kernels | concise author-facing orchestration code with small arg builders and runtime-owned lowering | runtime is platform-specific and owns more scheduling machinery, but the orchestration-surface simplicity is useful evidence for Megacu's thin-core direction |

## Audience

- GPU kernel engineers writing CUDA, CuTe, CUTLASS, Triton-like kernels, or
  future platform-native kernels who want in-kernel dependency composition
  without adopting a full compiler stack.
- Distributed training and inference engineers who need to combine compute,
  communication, events, and resource partitioning while preserving exact
  backend control.
- Compiler/runtime researchers who want a small target substrate for
  Event-Tensor-like dependency lowering.
- Backend authors who want to expose NVSHMEM, MSCCL++, CUDA P2P, multimem, or
  future transports through operation contracts rather than a new transport
  runtime.
- Framework integrators building PyTorch, SGLang, vLLM-style serving, or
  training runtimes who need stable descriptor/resource contracts and
  framework-visible launch-resource hooks without requiring Megacu itself to
  ship a Python package or tracing compiler.

## Meaningful Claim

Megacu is a zero-overhead, device-native composition layer for megakernels: it
makes task/event dependencies, memory ordering, participation policy, and backend
operation requirements explicit while letting expert authors keep
platform-native code in the hot path.

This is valuable because the current landscape is splitting in two directions.
The strongest systems are either vertical mega-kernel products or compiler
abstractions. Megacu should fill the missing middle: a small, inspectable
library/ABI layer for people building those systems by hand or lowering into
native kernels.

The layer must also be integrable. A minimal C++ substrate is only useful if
high-level systems can feed it tensor metadata, routing tables, workspace
buffers, and launch descriptors without hidden allocations or framework-specific
logic in the device hot path.
