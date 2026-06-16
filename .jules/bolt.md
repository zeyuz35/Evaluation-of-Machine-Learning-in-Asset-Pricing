## 2024-06-16 - Vectorize Array Slicing
**Learning:** Manual loops that iterate over an entire array dimension to assign values (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) are slower than direct vectorized slice assignments in R.
**Action:** Use native R vectorization (e.g., `X[, 1, index] <- 0`) to modify whole slices of arrays or matrices to improve execution speed, particularly inside computationally intensive loops.
