## 2024-05-23 - Optimize iterative rbind() loops in R Markdown files
**Learning:** O(N^2) "growing objects" bottlenecks are common when repeatedly calling `rbind()` inside `for` loops in R, leading to poor performance as objects are iteratively copied.
**Action:** Replace `for` loops containing `rbind()` with list accumulation (e.g., using `lapply()` to return individual data frames) followed by a final `do.call(rbind, ...)` to improve execution speed and memory management. Ensure you identify all occurrences across similar files (e.g., model fitting files, simulation functions).
