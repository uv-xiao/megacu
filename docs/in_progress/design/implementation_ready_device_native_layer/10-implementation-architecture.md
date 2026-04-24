# Implementation Architecture

This chapter records implementation guardrails that cut across the component
chapters. They are not new public concepts.

## Component Dependency Rules

The first implementation should keep these dependency boundaries:

```text
include/megacu/* public headers
include/megacu/detail/* record declarations

src/program              -> detail records
src/dispatcher           -> detail records
src/scheduler            -> detail records
src/platform/cuda        -> public CUDA views + detail launch records
src/backends/nvshmem     -> public NVSHMEM views + detail backend records
src/lowering             -> detail records + selected component plans
src/target               -> linked metadata + runtime validation + run entry
examples and tests       -> public APIs + compiled targets
```

Rules:

- public headers may name typed tags, public views, and builder APIs;
- public headers must not include CUDA/NVSHMEM headers except under explicit
  platform or backend headers;
- `detail` headers may define build-time and runtime metadata records, but they
  are not public authoring APIs;
- dispatcher code must not include scheduler implementation headers;
- scheduler code must not include backend implementation headers;
- lowering code may consume dispatcher, scheduler, platform, and backend plans;
- platform/backend adapters may provide validation and primitive bindings, but
  they must not choose dispatcher or scheduler policy;
- examples and integrations depend on the built target, never on private
  materializer internals.

## Metadata Representations

Megacu needs two metadata representations.

### Build-Time Records

Build-time records are host-only C++ objects used by the materializer. They may
use ergonomic C++ storage such as `std::vector`, `std::span`, `std::string_view`,
and type traits as long as their lifetimes are owned by the materializer.

Examples:

- `program_ir`
- `dispatch_plan`
- `schedule_plan`
- `kernel_plan`
- `backend_plan`

### Runtime Metadata Blob

The `.megacu.bin` runtime blob must be process-independent data. It must not
contain:

- raw pointers;
- `std::span`;
- `std::string_view`;
- `std::type_index`;
- host function pointers;
- C++ object vtables;
- allocator-owned container state.

The runtime blob must use fixed-width scalar records, offsets, counts, and enum
values. It must include:

- magic value and metadata ABI version;
- endianness and pointer-size marker;
- target id and component ids;
- offsets/counts for each table;
- checksum or size validation for the whole blob;
- feature bits for optional payloads such as multimem reduce.

`target_metadata(.json)` is the human/test inspection view. `.megacu.bin` is
the linked runtime view. Tests must compare both enough to prove they describe
the same target.

## Metadata Embedding

The materializer writes `.megacu.bin`. CMake embeds that binary as a native
object or equivalent linked data section. This is allowed because it is data
embedding, not C++/CUDA source generation.

The compiled target should access metadata through a narrow API:

```cpp
namespace megacu::detail {
struct target_id {
  std::uint64_t value;
};

struct target_metadata_view {
  std::span<const std::byte> bytes;
};

target_metadata_view linked_target_metadata(target_id id);
}
```

`target_metadata_for<Program, Components>()` may be a typed wrapper around the
linked metadata lookup, but it must not require generated C++ source. The first
implementation can make CMake create one linked binary-object target per
orchestrate target and expose a symbol pair for start/end addresses.

Runtime metadata validation must run before any kernel launch and return
non-OK `megacu::status` on invalid magic, version, size, checksum, or required
table absence.

`target_metadata_view` is a host-side view over linked bytes. The bytes it
points to are the serialized ABI; the `std::span` object itself is never stored
inside `.megacu.bin`.

## Status ABI

The core runtime ABI returns `megacu::status`.

First shape:

```cpp
namespace megacu {
enum class status_code : std::uint8_t {
  ok,
  invalid_argument,
  unsupported,
  backend_error,
  launch_error,
  metadata_error
};

struct status {
  status_code code;
  std::uint16_t detail;
  char const *message;
};
}
```

Rules:

- `message` must point to static storage or be null;
- core compiled targets must not return pointers into temporary strings;
- framework adapters may translate non-OK status values into framework
  exceptions at the adapter boundary;
- status detail values are component-owned small integers documented beside the
  component that emits them.

## Op Implementation ABI

CMake `OPS` entries must resolve to concrete implementation symbols before
target lowering succeeds.

Op symbols are role-based. Megacu must not require every op to provide both a
launchable CUDA kernel and a device-callable body.

First build-time op implementation record:

```cpp
namespace megacu::detail {
enum class op_entrypoint_role : std::uint8_t {
  launchable_kernel,
  callable_body,
  lowering_trampoline
};

struct op_entrypoint {
  op_entrypoint_role role;
  char const *symbol;
};

struct op_impl_entry {
  std::uint16_t op_slot;
  std::string_view op_name;
  std::span<const op_entrypoint> entrypoints;
};
}
```

Rules:

- `launchable_kernel` is a CUDA `__global__` kernel or equivalent
  platform-native launchable entrypoint; it is not host implementation code;
- separate-launch or multi-kernel lowering requires a `launchable_kernel`
  entrypoint;
- stitched single-persistent-kernel lowering requires a `callable_body` or a
  `lowering_trampoline` supplied by the lowering component;
- a target may provide only `launchable_kernel` entrypoints when the selected
  lowering does not need device-callable bodies;
- missing entrypoint roles required by the selected lowering fail target
  lowering;
- runtime metadata stores op slots and symbol ids, not raw function pointers;
- linked artifact inspection must prove that required op symbols are present in
  the built target.

## Error Boundaries

Implementation code should fail at the earliest owner:

- program builder errors: duplicate tags, invalid resource/event declarations;
- CMake target errors: missing op symbol, missing envelope, invalid component
  selection;
- materializer errors: unsupported event scope, invalid schedule guard, invalid
  backend requirements;
- runtime validation errors: bad problem shape, bad launch/team/session/views,
  incompatible metadata;
- device errors: kernel launch failure or backend primitive failure reported
  through the host status path when possible.

Device assertions are debugging aids only. They are not the primary validation
contract for host-visible misuse.
