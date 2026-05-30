## 2024-05-30 - Replace slow loops for vectorization
**Learning:** Found explicit loop for slice assignment `for (t in 1:dim(X_array_test)[1]) { X_array_test_zero[t, 1, index] <- 0 }` inside a `foreach` loop. Explicit iteration in R is notoriously slow compared to native vectorized subset assignment.
**Action:** Replace `for` loop with vectorized syntax `X_array_test_zero[, 1, index] <- 0` to dramatically improve memory manipulation performance in critical execution paths.
