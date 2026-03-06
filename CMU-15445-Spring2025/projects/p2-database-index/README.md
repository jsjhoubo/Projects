# P2 – Database Index (B+ Tree)

**Course:** CMU 15-445/645 Spring 2025  
**Topic:** Database Index  
**Release:** Feb 10, 2025  
**Due:** Mar 9, 2025  
**Spec:** https://15445.courses.cs.cmu.edu/spring2025/project2/

---

## Overview

Implement a **B+ Tree** index that stores key-value pairs on disk using the Buffer Pool Manager from P1. The B+ tree must support concurrent access via a latch-crabbing (lock-crabbing) protocol.

### Tasks

1. **B+ Tree Pages** — Implement `BPlusTreeInternalPage` and `BPlusTreeLeafPage`.
2. **B+ Tree Operations** — Implement `Search`, `Insert`, and `Delete` with proper page splits and merges/redistribution.
3. **Concurrent Index** — Add latch crabbing for thread-safe concurrent reads and writes.
4. **Index Iterator** — Implement a leaf-page iterator for sequential scans.

## Key Classes

| Class | File |
|-------|------|
| `BPlusTreeInternalPage` | `src/include/storage/page/b_plus_tree_internal_page.h` |
| `BPlusTreeLeafPage` | `src/include/storage/page/b_plus_tree_leaf_page.h` |
| `BPlusTree` | `src/include/storage/index/b_plus_tree.h` |
| `IndexIterator` | `src/include/storage/index/index_iterator.h` |

## Build & Test

```bash
cd bustub/build
make -j$(nproc) b_plus_tree_insert_test b_plus_tree_delete_test b_plus_tree_concurrent_test
./test/b_plus_tree_insert_test
./test/b_plus_tree_delete_test
./test/b_plus_tree_concurrent_test
```

## Implementation Notes

- Use **latch crabbing**: acquire child latch before releasing parent latch; release parent early if child is "safe" (not full/half-full).
- Leaf pages maintain a linked list (`next_page_id`) for efficient range scans.
- Handle edge cases: tree is empty, root is a leaf, root splits, tree collapses to a single node.
- Always use `BUSTUB_ASSERT` for invariant checks during development.

## Files to Modify

```
src/include/storage/page/b_plus_tree_internal_page.h
src/storage/page/b_plus_tree_internal_page.cpp
src/include/storage/page/b_plus_tree_leaf_page.h
src/storage/page/b_plus_tree_leaf_page.cpp
src/include/storage/index/b_plus_tree.h
src/storage/index/b_plus_tree.cpp
src/include/storage/index/index_iterator.h
src/storage/index/index_iterator.cpp
```
