# P4 – Concurrency Control

**Course:** CMU 15-445/645 Spring 2025  
**Topic:** Concurrency Control  
**Release:** Mar 31, 2025  
**Due:** Apr 20, 2025  
**Spec:** https://15445.courses.cs.cmu.edu/spring2025/project4/

---

## Overview

Implement **transaction management** with concurrency control guarantees in BusTub. This project adds ACID transaction support on top of the query execution layer from P3.

### Tasks

1. **Lock Manager** — Implement table-level and row-level locks with shared/exclusive/intention lock modes supporting multiple isolation levels.
2. **Deadlock Detection** — Background thread running a cycle-detection algorithm (DFS) on the waits-for graph; abort the youngest transaction in each cycle.
3. **Concurrent Query Execution** — Add lock acquisition/release to executors (`SeqScan`, `Insert`, `Update`, `Delete`) following Two-Phase Locking rules.

## Lock Modes

| Mode | Symbol | Compatible With |
|------|--------|----------------|
| Intention Shared | IS | IS, IX, S, SIX |
| Intention Exclusive | IX | IS, IX |
| Shared | S | IS, S |
| Shared Intention Exclusive | SIX | IS |
| Exclusive | X | (none) |

## Isolation Levels

| Level | Behavior |
|-------|----------|
| `READ_UNCOMMITTED` | No shared locks on reads |
| `READ_COMMITTED` | S locks released immediately after read |
| `REPEATABLE_READ` | S locks held until commit (Strict 2PL) |

## Key Classes

| Class | File |
|-------|------|
| `LockManager` | `src/include/concurrency/lock_manager.h` |
| `TransactionManager` | `src/include/concurrency/transaction_manager.h` |
| `Transaction` | `src/include/concurrency/transaction.h` |

## Build & Test

```bash
cd bustub/build
make -j$(nproc) lock_manager_test transaction_test
./test/lock_manager_test
./test/transaction_test
```

## Implementation Notes

- Use a `std::condition_variable` in each lock request queue so waiting threads sleep until the lock is granted.
- The deadlock detection thread should run every 50ms (configurable).
- Aborting a transaction: set its state to `ABORTED`, throw `TransactionAbortException`.
- Remember to upgrade locks (IS → S, IX → X, etc.) when the same transaction requests a stronger mode.
- For `REPEATABLE_READ`: release all locks only at `Commit`/`Abort` time.

## Files to Modify

```
src/include/concurrency/lock_manager.h
src/concurrency/lock_manager.cpp
src/concurrency/transaction_manager.cpp
src/execution/seq_scan_executor.cpp
src/execution/insert_executor.cpp
src/execution/update_executor.cpp
src/execution/delete_executor.cpp
```
