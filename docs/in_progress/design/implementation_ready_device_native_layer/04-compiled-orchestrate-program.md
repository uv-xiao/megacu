# Compiled Orchestrate Program

The primary runtime story should be simple:

- author an orchestrate program in C++;
- let CMake compile/link it against Megacu component targets;
- call that compiled program directly.

That is the normal path. It is not a plugin loader story.

## Direct Call Surface

The normal runtime surface should look like:

```cpp
struct gemm_allreduce_overlap_program;
struct gemm_ar_workspace_slot;
struct event_storage_slot;
struct m_tiles_extent;
struct n_tiles_extent;

// Ordinary parameterized function exported by the compiled orchestrate target.
// It is the user/framework call surface; there is no public executor object.
megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

The function parameters fill the descriptor slots by type:

- `workspace` fills `gemm_ar_workspace_slot`;
- `events` fills `event_storage_slot`;
- `launch` fills the CUDA platform launch slot;
- `team` fills the backend handle slot;
- `problem` fills matrix shape, stride, datatype envelope, and tile extents.

That binding is linked/materialized inside the compiled target. It is not a
public `exec.bind(...)` step and does not require emitted C++/CUDA source.

The descriptor compiled into that target is the GEMM+AllReduce descriptor shown
in `01-program.md` and `06-examples.md`. The important properties are:

- no string path loading;
- no generic `runtime_env`;
- no generic resource lookup by string;
- no event lookup by string;
- no separate generated wrapper required for the primary API;
- explicit typed arguments in the function signature;
- explicit typed tags for domains, participants, events, and ops.

## First ABI Types

The first example should define these concrete view types under
`examples/cuda_nvshmem_gemm_allreduce/` and move reusable pieces into public
headers only after the implementation proves the shape:

```cpp
struct gemm_ar_problem {
  std::int32_t m;
  std::int32_t n;
  std::int32_t k;
  std::int32_t lda;
  std::int32_t ldb;
  std::int32_t ldp;
  std::int32_t ldc;
  std::int32_t tile_m;
  std::int32_t tile_n;
  std::int32_t tile_k;
  megacu::dtype dtype;
};

struct gemm_ar_workspace {
  megacu::tensor_view a;
  megacu::tensor_view b;
  megacu::symmetric_tensor_view partial;
  megacu::tensor_view c;
  megacu::span<std::byte> scratch;
};
```

The first view types only need:

```cpp
namespace megacu {
enum class status_code : std::uint8_t {
  ok,
  invalid_argument,
  backend_error
};

struct status {
  status_code code;
  std::string_view message;
};

struct tensor_view {
  void *data;
  std::int64_t bytes;
  dtype type;
};

struct backend_id {
  std::uint16_t value;
};

struct session_id {
  std::uint64_t value;
};

struct symmetric_buffer_view {
  void *data;
  std::int64_t bytes;
  backend_id backend;
  session_id session;
};

struct symmetric_tensor_view {
  symmetric_buffer_view buffer;
  dtype type;
};

struct event_storage_view {
  symmetric_buffer_view buffer;
};
}
```

Shape and stride live in `gemm_ar_problem` for this target. A general tensor
shape system is not required for the first implementation.

The first CUDA launch view only needs:

```cpp
namespace megacu::cuda {
struct launch_view {
  cudaStream_t stream;
  int device_ordinal;
};
}
```

The MPI or Torch launch adapter constructs this value. The compiled target only
validates and consumes it.

## Function Body Shape

The orchestrate function is ordinary C++ in the target. It may be handwritten in
`examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce_orchestrate.cc`:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  auto *metadata = megacu::detail::target_metadata_for<
      gemm_allreduce_overlap_program,
      cuda_nvshmem_static_components>();

  megacu::detail::runtime_slots slots;
  slots.set<gemm_ar_workspace_slot>(workspace);
  slots.set<event_storage_slot>(events);
  slots.set_platform(launch);
  slots.set_backend(team);
  slots.set_extent<m_tiles_extent>(ceil_div(problem.m, problem.tile_m));
  slots.set_extent<n_tiles_extent>(ceil_div(problem.n, problem.tile_n));
  slots.set_problem(problem);

  return megacu::detail::run_static_persistent(metadata, slots);
}
```

This skeleton is normative for implementation intent, not an exact API freeze.
The invariants are:

- the function receives all runtime values through typed parameters;
- it fills slots by typed tags;
- it computes dynamic extents from `problem`;
- it calls one linked execution entrypoint;
- it returns `megacu::status` for validation, launch, and backend failures;
- it does not choose strategy, parse names, compile, or load plugins.

## What Runs

The compiled target contains the user-authored orchestrate function, linked
Megacu/backend implementations, and compact target metadata.

At runtime:

1. deployment code calls `cuda_nvshmem_gemm_allreduce_overlap_orchestrate(...)`;
2. the compiled target receives explicit runtime values such as workspace
   views, event storage, `megacu::cuda::launch_view`,
   `megacu::nvshmem::team_view`, and `gemm_ar_problem`;
3. linked target code and metadata attach those values to pre-lowered slots and
   enter the selected CUDA/NVSHMEM execution path;
4. internal fast-path execution launches the selected kernels and returns
   `megacu::status`;
5. kernels use `kernel_context` to resolve current output-tile points, virtual
   participants, rank/team participants, and event endpoints.

This is still direct C++ execution. The target does not load itself by path and
does not choose a new backend or schedule at runtime.

## Internal Lowering And Metadata

The build graph may still materialize internal prepared data such as:

- lowered submission records;
- resource/view tables;
- event wait/signal tables;
- participant mapping tables;
- dispatcher payload;
- scheduler payload;
- kernel launch payload;
- backend/platform payload.

Those are target internals. They are not the normal user authoring surface.
