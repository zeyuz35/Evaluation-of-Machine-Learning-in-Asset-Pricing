## 2024-05-18 - Base R Subsetting for Faster Iterations
**Learning:** Using `dplyr::filter()` inside long-running `for` loops and `foreach` constructs introduces significant overhead. The same row subsetting operation can be performed much faster using base R matrix/data.frame subsetting like `dataset[dataset$time == time_periods[t], ]`.
**Action:** When working on performance refactors inside tightly constrained iterative environments in R, prefer base R subset operations over dplyr verbs when filtering by a simple condition.
