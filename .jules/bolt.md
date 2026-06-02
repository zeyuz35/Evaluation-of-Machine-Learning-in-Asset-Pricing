## 2024-05-19 - Vectorize explicitly looped multidimensional array assignment
**Learning:** `R/simulation/Simulation Models.Rmd` uses an explicit `for` loop over `t` to zero out parts of an array slice:
```R
    for (t in 1:dim(X_array_test)[1]) {
      X_array_test_zero[t, 1, index] <- 0
    }
```
In R, this is an inefficient O(N) pattern. Array indexing supports fully vectorized multi-dimensional slices. We can just use `X_array_test_zero[, 1, index] <- 0` to achieve exactly the same outcome natively and instantly, completely eliminating the loop.
**Action:** Replace `for` loop indexing with vectorized assignment to improve performance, following the `bolt` agent rule: "In R, to maximize performance, prefer native array and matrix vectorization over manual iterative subsets. Replace explicit nested loops iterating over dimension boundaries (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) with direct vectorized slice assignments (e.g., `X[, 1, index] <- 0`)."
