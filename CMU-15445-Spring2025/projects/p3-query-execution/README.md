# P3 – Query Execution

**Course:** CMU 15-445/645 Spring 2025  
**Topic:** Query Execution  
**Release:** Mar 10, 2025  
**Due:** Mar 30, 2025  
**Spec:** https://15445.courses.cs.cmu.edu/spring2025/project3/

---

## Overview

Implement query execution operators in BusTub using the **Volcano/Iterator model**. Each operator exposes a `Next()` method that returns the next output tuple, pulling from its child operator(s).

### Tasks

1. **Access Methods** — `SeqScan`, `IndexScan`
2. **Modification Operators** — `Insert`, `Update`, `Delete`
3. **Aggregation** — `Aggregation` (SUM, COUNT, MIN, MAX, GROUP BY)
4. **Joins** — `NestedLoopJoin`, `HashJoin`
5. **Sort & Limit** — `Sort`, `Limit`, `TopN`
6. **Query Optimization** — Implement simple rule-based optimizations (e.g., predicate push-down, NLJ → Hash Join).

## Key Classes

| Operator | File |
|----------|------|
| `SeqScanExecutor` | `src/execution/seq_scan_executor.cpp` |
| `IndexScanExecutor` | `src/execution/index_scan_executor.cpp` |
| `InsertExecutor` | `src/execution/insert_executor.cpp` |
| `UpdateExecutor` | `src/execution/update_executor.cpp` |
| `DeleteExecutor` | `src/execution/delete_executor.cpp` |
| `AggregationExecutor` | `src/execution/aggregation_executor.cpp` |
| `NestedLoopJoinExecutor` | `src/execution/nested_loop_join_executor.cpp` |
| `HashJoinExecutor` | `src/execution/hash_join_executor.cpp` |
| `SortExecutor` | `src/execution/sort_executor.cpp` |
| `LimitExecutor` | `src/execution/limit_executor.cpp` |
| `TopNExecutor` | `src/execution/topn_executor.cpp` |

## Build & Test

```bash
cd bustub/build
make -j$(nproc) executor_test
./test/executor_test
# Also run the BusTub shell for manual testing:
make -j$(nproc) shell
./bin/bustub-shell
```

## Implementation Notes

- All executors are initialized in `Init()` and produce tuples one-at-a-time in `Next()`.
- The `ExecutorContext` provides access to the catalog, buffer pool manager, and transaction.
- For `HashJoin`: build phase on the right/inner table, probe phase on left/outer table.
- For `Aggregation`: use a hash table keyed by `GROUP BY` columns.
- `TopN` should use a priority queue (heap) instead of sorting then limiting.

## Files to Modify

```
src/execution/seq_scan_executor.cpp
src/execution/index_scan_executor.cpp
src/execution/insert_executor.cpp
src/execution/update_executor.cpp
src/execution/delete_executor.cpp
src/execution/aggregation_executor.cpp
src/execution/nested_loop_join_executor.cpp
src/execution/hash_join_executor.cpp
src/execution/sort_executor.cpp
src/execution/limit_executor.cpp
src/execution/topn_executor.cpp
src/optimizer/optimizer_custom_rules.cpp
```
