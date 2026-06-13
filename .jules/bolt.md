## 2024-06-13 - Replace iterative assignment with vectorized slice assignment in R
**Learning:** Manual loops that iterate over an entire dimension of an array (e.g., `for (t in 1:dim(X)[1]) { X[t, 1, index] <- 0 }`) are much slower than direct vectorized slice assignments in R.
**Action:** Replace `for` loops across tensor dimensions with vectorized subsetting (e.g., `X[, 1, index] <- 0`) whenever performing whole-slice assignments in R to boost execution speed.
