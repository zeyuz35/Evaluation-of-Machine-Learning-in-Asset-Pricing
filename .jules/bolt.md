## 2024-06-13 - Replace explicit nested loops iterating over dimension boundaries with direct vectorized slice assignments
**Learning:** To maximize performance in R, native array and matrix vectorization should be used over manual iterative subsets. Explicit nested loops iterating over dimension boundaries (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) can be replaced with direct vectorized slice assignments (e.g., `X[, 1, index] <- 0`).
**Action:** Always search for explicit iterative subsets using `dim()` or `length()` and replace them with base R vectorized subsets.
