## 2024-05-17 - [Vectorize iterative rbind in loops]
**Learning:** In R, iteratively calling `rbind()` inside a `for` loop forces memory reallocation on every iteration, leading to $O(N^2)$ time complexity. This is a common performance anti-pattern.
**Action:** Replace `rbind()` inside loops with list accumulation using `lapply()`, followed by a single `do.call(rbind, ...)` outside the loop to dramatically speed up execution and reduce memory overhead.
