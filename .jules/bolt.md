## 2024-05-18 - Replacing iterative array slicing with direct vectorization
**Learning:** Found a loop iterating over array dimensions (`for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) instead of utilizing vectorization (`X[, 1, index] <- 0`).
**Action:** Replace explicit nested loops iterating over dimension boundaries with direct vectorized slice assignments.
