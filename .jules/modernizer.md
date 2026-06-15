## 2024-06-15 - Vectorizing manual slice assignments in LSTM variable importance

**Learning:** R native vectorized subsetting is both cleaner and faster than a `for` loop over dimensions (e.g. `X[t, 1, index] <- 0` in a `for` loop over `t` vs `X[, 1, index] <- 0`).
**Action:** Replace `for (t in 1:dim(X_array_test)[1]) { X_array_test_zero[t, 1, index] <- 0 }` with `X_array_test_zero[, 1, index] <- 0`.
