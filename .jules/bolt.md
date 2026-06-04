## 2024-06-04 - Vectorizing Array Assignments
**Learning:** R performs poorly with explicit loops iterating over dimension boundaries of matrices and arrays (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`), especially when nested inside other loops. Vectorization using subsetting (`X[, 1, index] <- 0`) takes advantage of compiled C code within R, significantly improving execution speed.
**Action:** Replace manual iterative subsets and element-wise loops with direct vectorized slice assignments whenever operating on arrays or matrices in R.
