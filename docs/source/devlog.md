## [19-09-04]
#### Added
- stacktrace support and handler for segfault.
- option to use memory santilizer (-DUSE_MEMCHECK=on) to check memory errors, off by default

#### Bug 
- tensor_destroy(&tmp), which is used as x in conv layer and attached to cache.

#### TODO
1. I need to allocate those buff in the beginning, work buffer!
2. I shall remove the list for parameters so that they can be mapped without manually allocate them.
  - I shall co-design this with numa support.
  - "memory planner"
