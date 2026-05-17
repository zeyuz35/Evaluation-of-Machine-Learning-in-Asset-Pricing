## Modernizer Journal
## 2024-05-28 - [Performance Fix for iterative rbind in loops]
**Learning:** In R, iteratively appending rows to a data frame or matrix inside a for loop using `rbind` causes O(N^2) memory reallocation (the "growing objects" problem), which degrades performance significantly.
**Action:** Replace iterative `rbind` within loops with list accumulation (e.g., using `lapply`) followed by a single `do.call(rbind, ...)` call to aggregate the results efficiently.
