## 2026-06-18 - Vectorize Array Slice Assignment
**Learning:** Manual for loops iterating over array dimensions (e.g., for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }) are computationally expensive in R. Direct native vectorized slice assignments (e.g., X[, 1, index] <- 0) are significantly faster and more concise.
**Action:** Replace for loop dimension iteration with direct vectorized slice assignment where possible to optimize memory and execution time.
