## 2024-05-15 - Vectorized array slice assignment in LSTM variable importance
**Learning:** Manual loops that iterate over an entire dimension of an array (e.g., `for (t in 1:dim(X_array_test)[1]) { X_array_test_zero[t, 1, index] <- 0 }`) can be replaced with direct native vectorized slice assignments (`X_array_test_zero[, 1, index] <- 0`).
**Action:** Replace the loop with the vectorized equivalent.
