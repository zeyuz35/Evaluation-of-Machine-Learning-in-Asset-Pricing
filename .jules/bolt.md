## 2024-05-24 - Array Slicing Optimization
**Learning:** Manual loops that iterate over an entire dimension of an array or matrix (e.g., `for (t in 1:dim(X)[1])`) to modify slices can be very slow in R.
**Action:** Replace them with direct native vectorized slice assignments (e.g., `X[, 1, index] <- 0`) which is significantly faster and cleaner.
