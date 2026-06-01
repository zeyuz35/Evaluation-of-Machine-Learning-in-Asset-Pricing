## Modernizer Journal
## 2026-06-01 - Vectorize array slice assignment
**Learning:** Nested loops iterating over array/matrix dimension boundaries (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) are slow and less readable.
**Action:** Replace explicit loop dimensions with direct vectorized slice assignments (e.g., `X[, 1, index] <- 0`) to improve performance and code readability natively.
