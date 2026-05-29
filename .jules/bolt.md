## 2024-05-01 - Replace loop with vectorized array assignment in LSTM variable importance
**Learning:** In R, nested loops for array subset assignment (e.g. `X_array_test_zero[t, 1, index] <- 0`) are much slower than direct vectorization (`X_array_test_zero[, 1, index] <- 0`).
**Action:** Always prefer native R array and matrix vectorization over manual iteration along axes to improve performance.
