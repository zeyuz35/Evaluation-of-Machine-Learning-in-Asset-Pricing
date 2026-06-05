## 2024-05-23 - Native array vectorization
**Learning:** Manual loops over array dimension boundaries for scalar assignment (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) are much slower in R compared to native vectorized slice assignments (e.g., `X[, 1, index] <- 0`).
**Action:** Use vectorization directly when assigning elements or slices of multi-dimensional arrays instead of manually iterating over dimensions.
