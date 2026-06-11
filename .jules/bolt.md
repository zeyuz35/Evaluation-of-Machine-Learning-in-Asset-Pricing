## 2024-05-24 - Vectorize Array Assignment
**Learning:** Manual loop iteration (`for (t in 1:dim(X_array_test)[1])`) over tensor/array slices is slower than native vectorized assignment in R.
**Action:** Replace explicit loop slice assignments with direct vectorized subsetting (e.g., `X[, 1, index] <- 0`) to improve performance.
