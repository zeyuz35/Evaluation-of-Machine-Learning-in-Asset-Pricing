## Modernizer Journal
## 2024-05-29 - Vectorize Array Assignment over Iterative Substitution
**Learning:** Found explicit R loop iterating over dimension boundaries to substitute values in an array (`for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`). This creates unnecessary overhead.
**Action:** Replaced with direct vectorized slice assignments (`X[, 1, index] <- 0`) which base R implements natively in C, avoiding R loop interpretation and significantly improving runtime performance for large dimension arrays.
