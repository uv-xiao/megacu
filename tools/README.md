# Tools

Run scripts are organized by platform/backend pair and shared by default:

```text
tools/<platform>_<backend>/
```

Add `tools/<platform>_<backend>/<example>/` only when an example needs a unique
script. Scripts in this tree should be thin wrappers for building, running, or
validating examples under the matching `examples/<platform>_<backend>/` group.
Docker-backed scripts should use shared assets from the matching
`docker/<platform>_<backend>/` group unless an example-specific asset is
justified.
