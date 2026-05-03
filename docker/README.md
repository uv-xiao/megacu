# Docker Assets

Docker assets are organized by platform/backend pair and shared by default:

```text
docker/<platform>_<backend>/
```

Add `docker/<platform>_<backend>/<example>/` only when an example needs a unique
image or container asset. Shared Dockerfiles should support all examples under
the matching `examples/<platform>_<backend>/` group and be referenced by scripts
under the matching `tools/<platform>_<backend>/` group.
