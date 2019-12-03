## [19-12-01]
#### notes
1. all reduce now can be cooperated into blob itself
### TODO
* Use one blob function to copy initialize weights (same function can be used in all-reduce)
## [19-11-20]
#### Plan
1. thread model so that each layer could have different parallelsim.
2. vggnet b.
3. could use the tape variable can be used to  define how data is mapped!
  - layer_setup takes in policy(dp or mp)
  - each layer then constructs Blob: argument (nr_copies or nr_partitions), so we know how a blob is allocated
     * for example of dp, input/output will be partiioned, weight/bias blob will be duplicated
     * for example of mp, input/output will be duplicated, weight/bias will be partitioned.
     * between layer?
4. Notes:
  weight init after add hybrid
  also need to check all index data[0]

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
