# In-Progress Design Docs

This directory holds active design work that is not ready for long-term
architecture docs yet.

Use a short top-level entry point for each workstream, then keep the detailed
chapters in one ordered subdirectory when the topic has enough depth to need
multiple files.

## Active Workstreams

- `megacu_cpp_cuda_layer.md`: stable entry point for the Megacu device-native
  layer design.
- `megacu_device_native_layer/`: ordered chapter set for the active Megacu
  redesign. Read `00-overview.md` first, then continue numerically.

The numbered chapter files are the canonical design. Files in the `90+` range
are redesign notes and process records.
