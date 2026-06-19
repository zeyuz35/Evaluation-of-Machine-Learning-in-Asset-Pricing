## 2024-06-20 - Vectorize array assignments
**Learning:** In R, replacing nested loop dimension iterations (e.g. `for (i in 1:N) { for (t in 1:Time) { X[i, , t] <- ... } }`) with vectorized slice assignments (e.g., `for (t in 1:Time) { X[, , t] <- ... }`) improves execution speed dramatically without sacrificing readability.
**Action:** Always favor native vectorized array slice assignments over nested explicit scalar/vector loops.
