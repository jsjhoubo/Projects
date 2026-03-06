# P0 – C++ Primer

**Course:** CMU 15-445/645 Spring 2025  
**Topic:** C++ Primer  
**Release:** Jan 13, 2025  
**Due:** Jan 26, 2025  
**Spec:** https://15445.courses.cs.cmu.edu/spring2025/project0/

---

## Overview

A warm-up project to get familiar with modern C++ (C++17) features and the BusTub development environment. Key tasks typically include:

- Implementing a copy-on-write trie (CoW Trie) data structure
- Understanding move semantics, smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- Template programming and generic types
- Using `std::optional`, `std::string_view`, structured bindings
- Passing provided unit tests with `make -j$(nproc) p0_grader`

## Key C++ Concepts

| Concept | Notes |
|---------|-------|
| Smart pointers | `unique_ptr`, `shared_ptr` — no raw `new`/`delete` |
| Move semantics | `std::move`, rvalue references |
| `const` correctness | Mark methods `const` where appropriate |
| Templates | Generic `Get<T>` / `Put<T>` trie operations |
| `std::optional` | Return `std::nullopt` when key not found |

## Build & Test

```bash
cd bustub/build
make -j$(nproc) trie_test trie_noncopy_test
./test/trie_test
./test/trie_noncopy_test
```

## Files to Modify

```
src/include/primer/trie.h
src/primer/trie.cpp
```

## Notes

- Read the spec carefully — the trie nodes are **immutable** (copy-on-write semantics).
- Do **not** store `std::shared_ptr<TrieNode>` in multiple places without understanding ownership.
