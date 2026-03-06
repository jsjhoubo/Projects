# CMU 15-445/645: Intro to Database Systems (Spring 2025)

Course website: https://15445.courses.cs.cmu.edu/spring2025/

This directory tracks all homework assignments and programming projects for the CMU 15-445 Spring 2025 course. All programming projects are implemented in C++ using the [BusTub](https://github.com/cmu-db/bustub) educational DBMS.

---

## Schedule Overview

| # | Type | Topic | Release | Due |
|---|------|-------|---------|-----|
| HW1 | Homework | SQL | Jan 15, 2025 | Jan 29, 2025 |
| P0  | Project  | C++ Primer | Jan 13, 2025 | Jan 26, 2025 |
| P1  | Project  | Buffer Pool Manager | Jan 22, 2025 | Feb 9, 2025 |
| HW2 | Homework | Storage | Jan 29, 2025 | Feb 9, 2025 |
| HW3 | Homework | Indexes & Filters | Feb 12, 2025 | Feb 23, 2025 |
| P2  | Project  | Database Index | Feb 10, 2025 | Mar 9, 2025 |
| HW4 | Homework | Query Execution | Mar 12, 2025 | Mar 23, 2025 |
| P3  | Project  | Query Execution | Mar 10, 2025 | Mar 30, 2025 |
| HW5 | Homework | Concurrency Control | Mar 26, 2025 | Apr 6, 2025 |
| P4  | Project  | Concurrency Control | Mar 31, 2025 | Apr 20, 2025 |
| HW6 | Homework | Distributed Databases | Apr 8, 2025 | Apr 20, 2025 |

---

## Homework Assignments

Homework assignments are written/conceptual exercises covering the theory behind each topic.

- [HW1 – SQL](homework/hw1-sql/)
- [HW2 – Storage](homework/hw2-storage/)
- [HW3 – Indexes & Filters](homework/hw3-indexes-filters/)
- [HW4 – Query Execution](homework/hw4-query-execution/)
- [HW5 – Concurrency Control](homework/hw5-concurrency-control/)
- [HW6 – Distributed Databases](homework/hw6-distributed-databases/)

## Programming Projects

All programming projects use the **BusTub** DBMS codebase written in C++17.

- [P0 – C++ Primer](projects/p0-cpp-primer/)
- [P1 – Buffer Pool Manager](projects/p1-buffer-pool-manager/)
- [P2 – Database Index](projects/p2-database-index/)
- [P3 – Query Execution](projects/p3-query-execution/)
- [P4 – Concurrency Control](projects/p4-concurrency-control/)

---

## Environment Setup

```bash
# Clone the BusTub starter code (see course Gradescope for the private link)
git clone <bustub-repo> bustub
cd bustub

# Install dependencies (Ubuntu/Debian)
sudo apt-get install -y cmake gcc g++ clang clang-format valgrind

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Run tests
cd build
./test/<test_name>
```

---

## Progress

- [ ] HW1 – SQL
- [ ] P0  – C++ Primer
- [ ] P1  – Buffer Pool Manager
- [ ] HW2 – Storage
- [ ] HW3 – Indexes & Filters
- [ ] P2  – Database Index
- [ ] HW4 – Query Execution
- [ ] P3  – Query Execution
- [ ] HW5 – Concurrency Control
- [ ] P4  – Concurrency Control
- [ ] HW6 – Distributed Databases
