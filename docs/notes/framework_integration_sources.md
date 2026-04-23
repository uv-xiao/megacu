# Framework Integration Source Notes

- Date: 2026-04-23 Asia/Shanghai
- Purpose: inform Megacu's requirement that the minimal device-native layer be
  friendly to high-level frameworks such as PyTorch, SGLang, vLLM-style serving
  stacks, and future compiler/runtime frontends.
- Related design: `docs/design/megacu_cpp_cuda_layer.md`

## Sources Read

### FlashInfer

- Repository: `research/repos/flashinfer`
- Upstream: https://github.com/flashinfer-ai/flashinfer
- Local revision: `9f7adfb`, branch `main`
- Files read:
  - `README.md`
  - `docs/installation.rst`
  - `docs/vllm_routing_replay_integration.md`
  - `pyproject.toml`
  - `flashinfer/{decode.py,prefill.py,attention.py,mla/_core.py,utils.py}`
  - `flashinfer/jit/{core.py,cpp_ext.py,env.py}`
  - `flashinfer/{artifacts.py,__main__.py,api_logging.py,autotuner.py}`
  - `flashinfer/comm/{workspace_base.py,mapping.py}`

## What FlashInfer Contributes

FlashInfer is not a megakernel composition layer, but it is a strong example of
how low-level CUDA kernels can be made usable by PyTorch and serving frameworks
without forcing framework code into the device hot path.

Relevant integration patterns:

- Python package as the user entry point, with PyTorch tensors as the common
  data carrier.
- Optional split packages: core Python package, prebuilt cubins, and prebuilt JIT
  cache. This separates import/install ergonomics from kernel specialization
  and offline deployment.
- JIT module specs keyed by source, compile flags, CUDA architecture, and cache
  path. The source package can build on first use, but prebuilt artifacts can
  eliminate startup compilation.
- A plan/run split in wrappers. `plan()` prepares auxiliary metadata and is
  explicitly not part of CUDA Graph or `torch.compile` model logic; `run()` is
  the repeated hot path.
- User-owned workspace tensors. Wrappers accept preallocated device buffers,
  cache internal integer workspaces, and provide reset hooks instead of
  allocating unpredictably during every operation.
- CUDA Graph support through stable buffers and fixed shape/lifetime contracts.
  Megacu does not need to make CUDA Graph a core feature, but the useful lesson
  is that framework integration wants stable device buffers, stable descriptor
  addresses, and clear mutation/lifetime rules.
- Optional side-output buffers for framework-specific needs. The vLLM routing
  replay path accepts an optional preallocated output buffer; when it is absent,
  the kernel skips the write. The buffer can be larger than the active token
  count so serving code can reuse a max-size allocation.
- Dispatcher integration is treated carefully. FlashInfer has wrappers for
  PyTorch custom-op/fake-op registration, but comments note that
  `torch.library.custom_op` overhead is significant and the current helper can
  return identity decorators. The lesson is to make framework dispatch optional
  and measured, not mandatory for the fast path.
- Diagnostics are first-class: CLI commands show config, module status,
  downloaded artifacts, cache paths, CUDA/PyTorch versions, and compile command
  export. API logging can dump metadata and tensors under explicit environment
  control.
- Distributed/helper workspaces are explicit resources with lifecycle
  requirements. Wrappers warn when resources are not destroyed explicitly.

## Megacu Design Lessons

Megacu should not become a PyTorch operator library or a Python tracing
compiler. The shared layer should remain C++ and device-native. Framework
friendliness should instead be expressed as ABI and adapter requirements:

- framework adapters can build schedules from tensor metadata, graph/block
  tables, routing outputs, or serving batch metadata;
- host-side `plan` produces stable launch descriptors, event tensors, workspace
  requirements, and backend/platform launch packs;
- device-side `run` consumes only stable descriptors, typed tensor views,
  user-owned workspace, and concrete platform/backend handles;
- optional framework side outputs must be explicit descriptor fields and must be
  zero-cost when absent;
- descriptor and workspace lifetimes must be clear enough for graph capture,
  replay, async serving loops, and distributed framework runtimes;
- PyTorch/SGLang/vLLM adapters should live outside `shared` so no framework type
  enters event/task/schedule/kernel contracts;
- integration evidence should include Python import/build smoke tests, stable
  descriptor layout checks, no-allocation repeated-run checks, optional-output
  disabled-path inspection, and framework-adapter examples.

The key requirement is "thin but integrable": Megacu should expose the minimum
ABI and metadata needed by framework adapters, while all framework-specific
objects, dispatch registration, Python packaging, graph capture helpers, and
debug tooling remain outside the device hot path.
