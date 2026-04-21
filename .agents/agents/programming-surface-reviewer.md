# Programming Surface Reviewer Profile

Use this profile when reviewing public C++/CUDA APIs, examples, and tests.

Check:

- user code reads like normal CUDA/C++ rather than a heavy compiler DSL
- examples avoid constructor soup and raw payload blobs
- task, event, and schedule concepts are explicit but not verbose
- examples show realistic device code and host launch flow
- tests protect user-visible contracts, not only serialization

