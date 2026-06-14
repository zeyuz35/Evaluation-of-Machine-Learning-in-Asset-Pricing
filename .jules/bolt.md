## 2024-05-19 - Vectorize array assignments
**Learning:** In R, `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }` is slower than `X[, 1, index] <- 0`. Vectorized subset assignment is faster.
**Action:** Replace `for` loops used for subset assignments with native vectorized subsetting.
