## 2024-06-15 - Vectorize Array Slice Assignments
**Learning:** Manual loops iterating over an array dimension are slow in R.
**Action:** Use direct native vectorized slice assignments (e.g., `X[, 1, index] <- 0`) for better performance.
