# Tools

Example run scripts are organized by platform/backend pair and example name:

```text
tools/<platform>_<backend>/<example>/
```

Scripts in this tree should be thin wrappers for building, running, or
validating the matching example under `examples/`. Docker-backed scripts should
use Docker assets from the matching path under `docker/`.
