# P1 – Buffer Pool Manager

**Course:** CMU 15-445/645 Spring 2025  
**Topic:** Buffer Pool Manager  
**Release:** Jan 22, 2025  
**Due:** Feb 9, 2025  
**Spec:** https://15445.courses.cs.cmu.edu/spring2025/project1/

---

## Overview

Implement a thread-safe **Buffer Pool Manager (BPM)** for the BusTub DBMS. The BPM manages a fixed-size pool of frames in memory and handles fetching/evicting pages from disk.

### Components

1. **LRU-K Replacer** — Eviction policy tracking the K-th most recent access to each frame.
2. **Disk Scheduler** — Async disk I/O using a background worker thread and `std::promise`/`std::future`.
3. **Buffer Pool Manager** — Coordinates page fetch, pin/unpin, flush, and eviction.

## Key Classes

| Class | File |
|-------|------|
| `LRUKReplacer` | `src/include/buffer/lru_k_replacer.h` |
| `DiskScheduler` | `src/include/storage/disk/disk_scheduler.h` |
| `BufferPoolManager` | `src/include/buffer/buffer_pool_manager.h` |

## Build & Test

```bash
cd bustub/build
make -j$(nproc) lru_k_replacer_test buffer_pool_manager_test
./test/lru_k_replacer_test
./test/buffer_pool_manager_test
```

## Implementation Notes

- `LRUKReplacer::Evict()` should pick the frame whose K-th most recent access is the earliest (largest backward distance). If a frame has fewer than K accesses, it has infinite backward distance.
- Use `std::mutex` / `std::shared_mutex` for thread safety.
- `DiskScheduler` uses a `Channel<DiskRequest>` (provided) for async requests.
- Pin count: a page with `pin_count > 0` must **not** be evicted.

## Files to Modify

```
src/include/buffer/lru_k_replacer.h
src/buffer/lru_k_replacer.cpp
src/include/storage/disk/disk_scheduler.h
src/storage/disk/disk_scheduler.cpp
src/include/buffer/buffer_pool_manager.h
src/buffer/buffer_pool_manager.cpp
```
