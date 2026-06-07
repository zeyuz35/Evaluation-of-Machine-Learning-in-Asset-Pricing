## 2024-05-24 - [Vectorized array assignment]
**Learning:** [In R, replacing explicit nested loops iterating over dimension boundaries with direct vectorized slice assignments maximizes performance.]
**Action:** [Use direct vectorized slice assignments (e.g., `X[, 1, index] <- 0`) instead of manual iterative subsets (e.g., `for(t in 1:dim(X)[1]) X[t, 1, index] <- 0`).]
